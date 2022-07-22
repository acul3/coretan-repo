import sys
import requests
import re
import argparse
from bs4 import BeautifulSoup
import time
import datetime


headers = {
    'cache-control': 'max-age=0',
    'user-agent': 'Mozilla/5.0',
    'accept': 'text/html,application/xhtml+xml,application/xml',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en,en-US;q=0.9,id;q=0.8,de;q=0.7,ms;q=0.67'
}


class Kompas():
    document_url = "https://{topic}.kompas.com/search/{date}"
    topics = [
        "nasional", "health", "edukasi", "money", "lifestyle", "tekno",
        "properti", "bola", "travel", "otomotif", "kolom"
    ]

    def __init__(self):
        pass

    @staticmethod
    def get_article_links(start_date, end_date):
        """
        Get all article links from Kompas website. The function returns a generator.
        :param start_date:
        :param end_date:
        :return:
        """
        start = datetime.date(*[int(t) for t in start_date.split("-")])
        end = datetime.date(*[int(t) for t in end_date.split("-")])
        one_day = datetime.timedelta(days=1)
        date_current = start
        date_diff = end - date_current
        while date_diff.days >= 0:
            print(f"Download {date_current}", end=" ", file=sys.stderr, flush=True)
            article_index = set()
            for topic in Kompas.topics:
                print(".", end="", file=sys.stderr, flush=True)
                date = f"{date_current.year}-{date_current.month:02d}-{date_current.day:02d}"
                url = Kompas.document_url.replace("{topic}", str(topic)).replace("{date}", str(date))
                try:
                    while True:
                        time_to_wait = 10
                        while True:
                            try:
                                req = requests.get(url, headers=headers)
                                break
                            except requests.exceptions.ConnectionError:
                                # Sleep time_to_wait seconds before trying again
                                time.sleep(time_to_wait)
                                time_to_wait *= 1.5
                        if req.url != url or req.status_code == 401:
                            return None
                        page = BeautifulSoup(req.content, 'html.parser')
                        links = page.select("h3 .article__link")
                        for link in links:
                            index = re.sub(r".+/read/\d+/\d+/\d+/(\d+).+", r"\1", link["href"])
                            if index not in article_index:
                                article_index.add(index)
                                yield {
                                    "url": link["href"],
                                    "title": link.text,
                                    "date": date,
                                    "topic": topic
                                }
                        next_page = page.select_one("a.paging__link--next:-soup-contains('Next')")
                        if next_page is not None:
                            url = next_page["href"]
                        else:
                            break

                except KeyError as ke:
                    print(ke)
                    return None
            date_current = date_current + one_day
            date_diff = end - date_current
            print("", file=sys.stderr, flush=True)

    @staticmethod
    def get_topics():
        return Kompas.topics


def main():
    """
    This function print out article links in csv format or save them to a couchbase database bucket.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", type=str, required=True,
                        help="Start date, i.e. 2019-01-10")
    parser.add_argument("-e", "--end_date", type=str, required=False,
                        help="End date (inclusive), i.e. 2019-01-20")
    parser.add_argument("-c", "--couchbase", required=False, action='store_true',
                        help="Store the links in couchbase")
    parser.add_argument("--host", type=str, required=False, default="localhost",
                        help="Hostname for couchbase")
    parser.add_argument("-u", "--username", type=str, required=False, default="user",
                        help="Username for couchbase")
    parser.add_argument("-p", "--password", type=str, required=False, default="password",
                        help="Password for couchbase")
    args = parser.parse_args()
    if args.end_date is None:
        args.end_date = args.start_date
    if args.couchbase:
        from couchbase_kompas import CouchBaseKompas
        couchbase = CouchBaseKompas(host=args.host, username=args.username, password=args.password)
    else:
        couchbase = None
    kompas = Kompas()
    article_links = kompas.get_article_links(args.start_date, args.end_date)
    for i, article in enumerate(article_links):
        if args.couchbase:
            couchbase.upsert_document(article)
        else:
            title = article["title"].replace('"', '\\"')
            print(f'"{article["date"]}","{article["topic"]}","{article["url"]}","{title}"')


if __name__ == '__main__':
    main()