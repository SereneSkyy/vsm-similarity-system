import requests
from bs4 import BeautifulSoup
import os
import time

# Deliberately mixed topics: some pairs closely related (should score HIGH
# similarity), some unrelated (should score near-ZERO) — this makes the
# similarity matrix meaningful to analyze in the report.
TOPICS = [
    "2026_FIFA_World_Cup",              # sports
    "2026_Winter_Olympics",             # sports (related to #1)
    "Artificial_intelligence",          # tech
    "ChatGPT",                          # tech (related to #3)
    "Climate_change",                   # environment
    "Renewable_energy",                 # environment (related to #5)
    "Cryptocurrency",                   # finance/tech
    "Inflation",                        # economics
]

BASE_URL = "https://en.wikipedia.org/wiki/"
OUTPUT_DIR = "docs"
HEADERS = {"User-Agent": "TECH400-Student-Assignment/1.0 (educational use)"}


def scrape_article(topic):
    url = BASE_URL + topic
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content_div.find_all("p")

    text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
    return text


def save_document(topic, text):
    filename = topic.replace("_", "-").lower() + ".txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {filepath} ({len(text)} characters)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for topic in TOPICS:
        try:
            text = scrape_article(topic)
            save_document(topic, text)
            time.sleep(1)
        except Exception as e:
            print(f"Failed to scrape {topic}: {e}")


if __name__ == "__main__":
    main()