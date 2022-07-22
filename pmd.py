# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import json
import os
from abc import abstractmethod
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import datasets
import pyarrow as pa
from datasets import StreamingDownloadManager, load_dataset

_CITATION = """
@inproceedings{singh2022flava,
  title={Flava: A foundational language and vision alignment model},
  author={Singh, Amanpreet and Hu, Ronghang and Goswami, Vedanuj and Couairon, Guillaume and Galuba, Wojciech and Rohrbach, Marcus and Kiela, Douwe},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15638--15650},
  year={2022}
}

"""

_DESCRIPTION = """
Introduced in FLAVA paper, Public Multimodal Dataset (PMD) is a collection of publicly-available image-text pairs datasets. PMD in total contains 70M image-text pairs with 68M unique images. The dataset contains pairs from Conceptual Captions, Conceptual Captions 12M, WIT, Localized Narratives, RedCaps, COCO, SBU Captions, Visual Genome and a subset of YFCC100M dataset.
"""

_HOMEPAGE = "https://flava-model.github.io/"

_LICENSE = """
Please refer to individual LICENSES of each datasets. Most of them should be under Creative Commons license but users should verify each image individually.
"""

_FEATURES = datasets.Features(
    {
        # Some images provide an url others provide an Image. Both are exclusive.
        "image_url": datasets.Value("string"),
        "image": datasets.Image(),
        # An image can have multiple texts associated with it, but we take a cartesan product
        # for PMD
        "text": datasets.Value("string"),
        # Define where the sample comes from, this is necessary when we start to use aggregated versions like PMD.
        "source": datasets.Value("string"),
        # We commit any kind of additional information in json format in `meta`
        "meta": datasets.Value("string"),
    }
)

_BASE_HF_URL = 'https://huggingface.co/datasets/munggok/pmd_indonesia/resolve/main/data/'


def _file_to_chunks(fi, chunk_size):
    chunk = tuple(itertools.islice(fi, chunk_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(fi, chunk_size))


def json_serializer(o):
    if isinstance(o, datetime):
        return str(o)

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


class LoaderInterface:
    @abstractmethod
    def _generate_batches(self):
        raise NotImplementedError()


class BaseLoader(LoaderInterface):
    def __init__(self, source: str, split: str, writer_batch_size: int, chunk_size: int = 1):
        self.source = source
        self.split = split
        self.writer_batch_size = writer_batch_size
        self.chunk_size = chunk_size
        self.gen_kwargs = {}

    @abstractmethod
    def _generate_examples(self, examples: List[Any], **kwargs) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _build_rows_iterator(self, chunk_size: int, **kwargs) -> Iterator[List[Any]]:
        raise NotImplementedError()

    @abstractmethod
    def _generate_tables(self, examples: List[Any], **kwargs) -> pa.Table:
        raise NotImplementedError()

    def _generate_batches(self):
        rows_iterator = self._build_rows_iterator(chunk_size=self.chunk_size, **self.gen_kwargs)
        if self.num_proc == 1:
            for row in rows_iterator:
                yield self._generate_tables(row, **self.gen_kwargs)
        else:
            assert self.num_proc > 1
            with Pool(self.num_proc) as pool:
                tables_iterator = pool.imap(
                    partial(self._generate_tables, **self.gen_kwargs),
                    rows_iterator,
                    chunksize=1,
                )
                for table in tables_iterator:
                    yield table


class DatasetsLoader(BaseLoader):
    """Helper as some datasets are already implemented"""

    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str],
        split: str,
        num_proc: int,
        datasets_batch_size: int = 1000,
        streaming: bool = False,
    ):
        super(DatasetsLoader, self).__init__(source=dataset_name, split=split, writer_batch_size=datasets_batch_size)
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.num_proc = num_proc
        self.datasets_batch_size = datasets_batch_size
        self.streaming = streaming

    @abstractmethod
    def cast_to_pmd_features(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Return list of caster rows. Casted row are either PMD features"""
        raise NotImplementedError()

    def _generate_examples(self, examples: List[Any], **kwargs) -> Dict[str, List[Any]]:
        batch = {}
        for key in examples[0]:
            batch[key] = []

        for example in examples:
            for key in example:
                batch[key].append(example[key])
        return batch

    def _generate_tables(self, examples: List[Any], **kwargs) -> pa.Table:
        output_batch = self.cast_to_pmd_features(self._generate_examples(examples, **kwargs))
        return pa.table(_FEATURES.encode_batch(output_batch))

    def _build_rows_iterator(self, chunk_size: int, **kwargs):
        dataset = load_dataset(self.dataset_name, self.config_name, split=self.split, streaming=self.streaming)
        buffer = []

        for row in dataset:
            buffer.append(row)
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        if len(buffer) > 0:
            yield buffer


class BaseLoaderWithDLManager(BaseLoader):
    """We use dl_manager to generate `gen_kwargs` needed in order to generate examples."""

    def __init__(
        self,
        dl_manager,
        source: str,
        split: str,
        num_proc: int,
        chunk_size: int,
        writer_batch_size: int = 10_000,
    ):
        super(BaseLoaderWithDLManager, self).__init__(source=source, split=split, writer_batch_size=writer_batch_size)
        self.gen_kwargs = self.generate_gen_kwargs(dl_manager)
        # Used for multiprocessing
        self.chunk_size = chunk_size
        self.num_proc = num_proc

    @abstractmethod
    def generate_gen_kwargs(self, dl_manager):
        raise NotImplementedError()

    def _generate_tables(self, examples: List[Any], **kwargs) -> pa.Table:
        return pa.table(_FEATURES.encode_batch(self._generate_examples(examples, **kwargs)))



class LocalizedNarrativesOpenImagesLoader(BaseLoaderWithDLManager):
    _ANNOTATION_URLs = {
        "train": _BASE_HF_URL + "ln_open_images_train_v6_captions_multi.jsonl",
        "validation": _BASE_HF_URL + "ln_open_images_validation_captions_multi.jsonl",
        "test": _BASE_HF_URL + "ln_open_images_test_captions_multi.jsonl",
    }

    def __init__(
        self,
        dl_manager,
        split: str,
        num_proc: int,
        chunk_size: int,
        writer_batch_size: int,
    ):
        super(LocalizedNarrativesOpenImagesLoader, self).__init__(
            dl_manager=dl_manager,
            source="localized_narratives__openimages",
            split=split,
            num_proc=num_proc,
            chunk_size=chunk_size,
            writer_batch_size=writer_batch_size,
        )

    def generate_gen_kwargs(self, dl_manager):
        annotation_file = dl_manager.download(self._ANNOTATION_URLs[self.split])
        return {"annotation_file": annotation_file, "split": self.split}

    def _build_rows_iterator(self, chunk_size: int, annotation_file: str, split: str) -> Iterator[List[Any]]:
        with open(annotation_file, "r", encoding="utf-8") as fi:
            yield from _file_to_chunks(fi, chunk_size)

    def _generate_examples(self, examples: List[Any], annotation_file: str, split: str) -> Dict[str, List[Any]]:
        annotations = [json.loads(line) for line in examples]

        # sanity check
        for annotation in annotations:
            assert "image_url" not in annotation

        return {
            "image_url": [
                f"https://s3.amazonaws.com/open-images-dataset/{split}/{annotation['image_id']}.jpg"
                for annotation in annotations
            ],
            "image": [None for _ in annotations],
            "text": [annotation["caption"] for annotation in annotations],
            "source": [self.source for _ in annotations],
            "meta": [
                json.dumps(
                    annotation,
                    default=json_serializer,
                    indent=2,
                )
                for annotation in annotations
            ],
        }


class LocalizedNarrativesCOCOLoader(BaseLoaderWithDLManager):
    _ANNOTATION_URLs = {
        "train": _BASE_HF_URL + "ln_coco_train_captions_multi.jsonl",
        "validation": _BASE_HF_URL + "ln_coco_val_captions_multi.jsonl",
    }
    _KARPATHY_CAPTION_URL = _BASE_HF_URL + "caption_datasets.zip"
    _IMAGES_URLS = {
        "train": "http://images.cocodataset.org/zips/train2017.zip",
        "validation": "http://images.cocodataset.org/zips/val2017.zip",
    }
    _SPLIT_MAP = {"train": "train2017", "validation": "val2017"}

    def __init__(
        self,
        dl_manager,
        split: str,
        num_proc: int,
        chunk_size: int,
        writer_batch_size: int,
    ):
        super(LocalizedNarrativesCOCOLoader, self).__init__(
            dl_manager=dl_manager,
            source="localized_narratives__coco",
            split=split,
            num_proc=num_proc,
            chunk_size=chunk_size,
            writer_batch_size=writer_batch_size,
        )

    def generate_gen_kwargs(self, dl_manager):
        annotation_file = dl_manager.download(self._ANNOTATION_URLs[self.split])
        image_folder = Path(dl_manager.download_and_extract(self._IMAGES_URLS[self.split]))
        karpathy_coco_file = Path(dl_manager.download_and_extract(self._KARPATHY_CAPTION_URL)) / "dataset_coco.json"

        with open(karpathy_coco_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
            invalid_images = {}
            for annotation in annotations["images"]:
                if annotation["split"] == "val" or annotation["split"] == "test":
                    invalid_images[int(annotation["cocoid"])] = 1

        return {
            "annotation_file": annotation_file,
            "base_image_path": image_folder / self._SPLIT_MAP[self.split],
            "invalid_images": invalid_images,
        }

    def _build_rows_iterator(
        self,
        chunk_size: int,
        annotation_file: str,
        base_image_path: Path,
        invalid_images: Dict[str, bool],
    ) -> Iterator[List[Any]]:
        with open(annotation_file, "r", encoding="utf-8") as fi:
            yield from _file_to_chunks(fi, chunk_size)

    def _generate_examples(
        self,
        examples: List[Any],
        annotation_file: str,
        base_image_path: Path,
        invalid_images: Dict[str, bool],
    ) -> Dict[str, List[Any]]:
        annotations = [json.loads(line.strip()) for line in examples]
        annotations = [line for line in annotations if int(line["image_id"]) not in invalid_images]

        return {
            "image_url": [None for _ in annotations],
            "image": [
                str((base_image_path / f"{annotation['image_id'].zfill(12)}.jpg")) for annotation in annotations
            ],
            "text": [annotation["caption"] for annotation in annotations],
            "source": [self.source for _ in annotations],
            "meta": [
                json.dumps(
                    annotation,
                    default=json_serializer,
                    indent=2,
                )
                for annotation in annotations
            ],
        }


class LocalizedNarrativesFlickr30kLoader(BaseLoaderWithDLManager):
    _LOCAL_IMAGE_FOLDER_NAME = "flickr30k-images.tar.gz"
    _ANNOTATION_URLs = {
        "train": _BASE_HF_URL + "ln_flickr30k_train_captions_multi.jsonl",
        "validation": _BASE_HF_URL + "ln_flickr30k_val_captions_multi.jsonl",
        "test": _BASE_HF_URL + "ln_flickr30k_test_captions_multi.jsonl",
    }

    def __init__(
        self,
        dl_manager,
        split: str,
        num_proc: int,
        chunk_size: int,
        writer_batch_size: int,
    ):
        super(LocalizedNarrativesFlickr30kLoader, self).__init__(
            dl_manager=dl_manager,
            source="localized_narratives__flickr30k",
            split=split,
            num_proc=num_proc,
            chunk_size=chunk_size,
            writer_batch_size=writer_batch_size,
        )

    @property
    def manual_download_instruction(self):
        return """\
            You need to go to http://shannon.cs.illinois.edu/DenotationGraph/data/index.html,
            and manually download the dataset ("Flickr 30k images."). Once it is completed,
            a file named `flickr30k-images.tar.gz` will appear in your Downloads folder
            or whichever folder your browser chooses to save files to.
            The dataset can then be loaded using the following command 
            `datasets.load_dataset("pmd", data_dir="<path/to/download_folder/flickr30k-images.tar.gz>")`.
            """

    def generate_gen_kwargs(self, dl_manager):
        if dl_manager.manual_dir is None:
            raise FileNotFoundError(
                f"Please set manual dir via `datasets.load_dataset('pmd', data_dir={{PATH}})` where `{{PATH}}` points to `{self._LOCAL_IMAGE_FOLDER_NAME}`.\n. Manual download instructions: {self.manual_download_instruction}"
            )
        image_dir = Path(dl_manager.manual_dir)
        annotation_file = dl_manager.download(self._ANNOTATION_URLs[self.split])
        return {"annotation_file": annotation_file, "archive": dl_manager.iter_archive(image_dir)}

    def _build_rows_iterator(self, chunk_size, annotation_file: str, archive: Path) -> Iterator[List[Any]]:
        annotations = {}
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                annotation = json.loads(line.strip())
                image_id = str(annotation["image_id"])
                if image_id not in annotations:
                    annotations[image_id] = []
                annotations[image_id].append(annotation["caption"])

        buffer = []
        for path, file in archive:
            if path.endswith(".jpg"):
                root, _ = os.path.splitext(path)
                imageid = os.path.basename(root)
                if imageid not in annotations:
                    continue
                captions = annotations[imageid]
                bytes = file.read()
                for caption in captions:
                    buffer.append(
                        {
                            "image": {"path": path, "bytes": bytes},
                            "caption": caption,
                        }
                    )
                    if len(buffer) == chunk_size:
                        yield buffer
                        buffer = []

        if len(buffer):
            yield buffer

    def _generate_examples(self, examples: List[Any], annotation_file: str, archive: Path) -> Dict[str, List[Any]]:
        return {
            "image_url": [None for _ in examples],
            "image": [annotation["image"] for annotation in examples],
            "text": [annotation["caption"] for annotation in examples],
            "source": [self.source for _ in examples],
            "meta": [
                json.dumps(
                    {"image": annotation["image"]["path"], "caption": annotation["caption"]},
                    default=json_serializer,
                    indent=2,
                )
                for annotation in examples
            ],
        }




class Conceptual12MLoader(DatasetsLoader):
    def __init__(self, split: str, num_proc: int, datasets_batch_size: int = 1000, **kwargs):
        super(Conceptual12MLoader, self).__init__(
            dataset_name="munggok/CC_12M_Indonesia",
            config_name=None,
            split=split,
            datasets_batch_size=datasets_batch_size,
            num_proc=num_proc,
            **kwargs,
        )

    def cast_to_pmd_features(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        metas = {k: v for k, v in batch.items() if k not in ["url", "caption"]}
        batch_size = len(next(iter(batch.values())))
        return {
            "image_url": batch["url"],
            "image": [None] * batch_size,
            "text": [caption for caption in batch["caption"]],
            "source": [self.source] * batch_size,
            "meta": [
                json.dumps(
                    {key: value[batch_id] for key, value in metas.items()},
                    default=json_serializer,
                    indent=2,
                )
                for batch_id in range(batch_size)
            ],
        }

class LaionIndo(DatasetsLoader):
    def __init__(self, split: str, num_proc: int, datasets_batch_size: int = 1000, **kwargs):
        super(LaionIndo, self).__init__(
            dataset_name="munggok/Laion_Indo",
            config_name=None,
            split=split,
            datasets_batch_size=datasets_batch_size,
            num_proc=num_proc,
            **kwargs,
        )

    def cast_to_pmd_features(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        metas = {k: v for k, v in batch.items() if k not in ["URL", "TEXT"]}
        batch_size = len(next(iter(batch.values())))
        return {
            "image_url": batch["URL"],
            "image": [None] * batch_size,
            "text": [caption for caption in batch["TEXT"]],
            "source": [self.source] * batch_size,
            "meta": [
                json.dumps(
                    {key: value[batch_id] for key, value in metas.items()},
                    default=json_serializer,
                    indent=2,
                )
                for batch_id in range(batch_size)
            ],
        }

class YFCC100MLoader(BaseLoaderWithDLManager):
    _ANNOTATION_URL = _BASE_HF_URL + "yfcc100m_subset.jsonl"

    def __init__(
        self,
        dl_manager,
        split: str,
        num_proc: int,
        chunk_size: int,
        writer_batch_size: int,
    ):
        super(YFCC100MLoader, self).__init__(
            dl_manager=dl_manager,
            source="yfcc100m",
            split=split,
            num_proc=num_proc,
            chunk_size=chunk_size,
            writer_batch_size=writer_batch_size,
        )
        self.chunk_size = chunk_size

    def generate_gen_kwargs(self, dl_manager):
        annotation_file = dl_manager.download(self._ANNOTATION_URL)
        return {"annotation_file": annotation_file}

    def _build_rows_iterator(self, annotation_file: str, chunk_size: int) -> Iterator[List[Any]]:
        with open(annotation_file, "r", encoding="utf-8") as f:
            buffer = []
            for line in f:
                buffer.append(json.loads(line))
                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []

            if len(buffer) > 0:
                yield buffer

    def _generate_examples(self, examples: List[Any], annotation_file: str) -> Dict[str, Any]:
        return {
            "image_url": [example["image_url"] for example in examples],
            "image": [example["image"] for example in examples],
            "text": [example["texts"][0] for example in examples],
            "source": [example["source"] for example in examples],
            "meta": [example["meta"] for example in examples],
        }


class PMDConfig(datasets.BuilderConfig):
    """BuilderConfig for PMD."""

    def __init__(
        self,
        name: str = "all",
        subset: Optional[str] = None,
        num_proc: Optional[int] = None,
        datasets_batch_size: int = 1000,
        sqlite3_batch_size: int = 10_000,
        chunk_size: int = 10_000,
        writer_batch_size: int = 10_000,
        use_flickr30k_ln: bool = False,
        **kwargs,
    ):
        if num_proc is None:
            # We disable multiprocessing.
            num_proc = 1
        super(PMDConfig, self).__init__(**kwargs)
        self.name = name
        self.subset = subset
        # determines how much we can load
        self.datasets_batch_size = datasets_batch_size
        self.sqlite3_batch_size = sqlite3_batch_size

        # Some datasets should be loaded via multiprocessing.
        self.num_proc = num_proc
        self.chunk_size = chunk_size

        # Batch writing
        self.writer_batch_size = writer_batch_size

        # LN
        self.use_flickr30k_ln = use_flickr30k_ln


class PMD(datasets.ArrowBasedBuilder):
    """Builder for Open Images subset of PMD."""

    BUILDER_CONFIG_CLASS = PMDConfig

    BUILDER_CONFIGS = [
        PMDConfig(name="all"),
        PMDConfig(name="localized_narratives"),
        PMDConfig(name="localized_narratives_openimages"),
        PMDConfig(name="localized_narratives_coco"),
        PMDConfig(name="conceptual_captions_12M"),
        PMDConfig(name="laion_indo"),
        PMDConfig(name="yfcc100M_subset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={"loaders": self._build_loaders(dl_manager, split_name)},
            )
            for split_name in [datasets.Split.TRAIN, datasets.Split.VALIDATION]
        ]

    def _build_loaders(self, dl_manager, split_name):
        loaders = []

        streaming = False
        if isinstance(dl_manager, StreamingDownloadManager):
            streaming = True

        if self.config.name in ["localized_narratives", "localized_narratives_openimages", "all"]:
            loaders.append(
                LocalizedNarrativesOpenImagesLoader(
                    dl_manager=dl_manager,
                    split=split_name,
                    num_proc=self.config.num_proc,
                    chunk_size=self.config.chunk_size,
                    writer_batch_size=self.config.writer_batch_size,
                )
            )

        if self.config.name in ["localized_narratives", "localized_narratives_coco", "all"]:
            loaders.append(
                LocalizedNarrativesCOCOLoader(
                    dl_manager=dl_manager,
                    split=split_name,
                    num_proc=self.config.num_proc,
                    chunk_size=self.config.chunk_size,
                    writer_batch_size=self.config.writer_batch_size,
                )
            )

        # Flickr30k is not by default a part of "all" as it requires manual download which is clunky
        if self.config.name in ["localized_narratives", "localized_narratives_flickr30k"]:
            if self.config.use_flickr30k_ln:
                loaders.append(
                    LocalizedNarrativesFlickr30kLoader(
                        dl_manager=dl_manager,
                        split=split_name,
                        num_proc=self.config.num_proc,
                        chunk_size=self.config.chunk_size,
                        writer_batch_size=self.config.writer_batch_size,
                    ),
                )

        # Rest of the datasets don't have other splits
        if split_name != datasets.Split.TRAIN:
            return loaders

        if self.config.name == "conceptual_captions_12M" or self.config.name == "all":
            loaders.append(
                Conceptual12MLoader(
                    split=split_name,
                    num_proc=self.config.num_proc,
                    datasets_batch_size=self.config.datasets_batch_size,
                    streaming=streaming,
                )
            )
        
        if self.config.name == "laion_indo" or self.config.name == "all":
            loaders.append(
                LaionIndo(
                    split=split_name,
                    num_proc=96,
                    datasets_batch_size=self.config.datasets_batch_size,
                    streaming=streaming,
                )
            )

        if self.config.name == "yfcc100M_subset" or self.config.name == "all":
            loaders.append(
                YFCC100MLoader(
                    dl_manager=dl_manager,
                    split=split_name,
                    num_proc=self.config.num_proc,
                    chunk_size=self.config.sqlite3_batch_size,
                    writer_batch_size=self.config.writer_batch_size,
                ),
            )
        return loaders

    def _generate_tables(self, loaders: List[BaseLoader]):
        idx = 0
        for loader in loaders:
            for elt in loader._generate_batches():
                yield idx, elt
                idx += 1
