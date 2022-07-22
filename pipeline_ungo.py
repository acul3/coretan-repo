import glob
import subprocess
import os
import shutil

file_wet_part = sorted(glob.glob('/data/wet_part/*'))
i = 9
print(file_wet_part)
for m in file_wet_part:
    print("download step")
    subprocess.call(["ungoliant","download",m,'1_download/','-t',"96"])
    print("asdasd")
    subprocess.call(["ungoliant", "pipeline",'1_download/','2_pipeline/'])
    files = glob.glob('/data/1_download/*')
    print('remove files downloads')
    for f in files:
        os.remove(f)
    print('renaming')
    name = 'id_meta_' + str(i) + '.jsonl'
    old_file = os.path.join("/data/2_pipeline/", "id_meta.jsonl")
    new_file = os.path.join("/data/meta_id/meta_auto", name)
    os.rename(old_file, new_file)
    files_pipe = glob.glob('/data/2_pipeline/*')
    i += 1
    for g in files_pipe:
        if os.path.isdir(g):
            shutil.rmtree(g,ignore_errors=True)
        else:
            os.remove(g)
    print('sekarang',i)

        