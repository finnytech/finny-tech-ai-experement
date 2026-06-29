import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json
import os
import argparse
import random
import time
import re
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("FinnyCrawlerV2")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
]

# Words that indicate a question/explanation structure
QUESTION_WORDS_EN = r"\b(how|why|what|who|where|which|can|should|is|are|does|do|did|explain|definition)\b"
QUESTION_WORDS_DE = r"\b(wie|warum|was|wer|wo|welche|kann|soll|ist|sind|erkläre|definition|bedeutung)\b"
QUESTION_PATTERN = re.compile(
    rf"({QUESTION_WORDS_EN}|{QUESTION_WORDS_DE})", 
    re.IGNORECASE
)

class FinnyKnowledgeCrawler:
    def __init__(self, seed_urls, output_file, max_pages=100000, max_concurrency=40, delay_per_domain=0.5):
        self.seed_urls = seed_urls
        self.output_file = output_file
        self.max_pages = max_pages
        self.max_concurrency = max_concurrency
        self.delay_per_domain = delay_per_domain
        
        self.visited = set()
        self.pages_scraped = 0
        
        self.domain_queues = {}
        self.domain_visited = {}
        self.last_fetch_time = {}
        self.active_domains = set()
        
        self.write_lock = asyncio.Lock()

        # Initialize seeds
        for url in self.seed_urls:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                if domain not in self.domain_queues:
                    self.domain_queues[domain] = asyncio.Queue()
                    self.domain_visited[domain] = set()
                self.domain_queues[domain].put_nowait(url)

    def extract_wikipedia_definition(self, soup, url):
        """Extracts concept definition from Wikipedia articles."""
        title_el = soup.find("h1", id="firstHeading")
        if not title_el:
            return None
        concept = title_el.get_text().strip()
        
        if ":" in concept and not concept.startswith("Category:"):
            return None

        body = soup.find("div", id="mw-content-text")
        if not body:
            return None
            
        paragraphs = []
        for child in body.find_all("p", recursive=True):
            p_text = child.get_text().strip()
            if len(p_text) > 50:
                paragraphs.append(p_text)
            if len(paragraphs) >= 3:
                break
                
        if not paragraphs:
            return None
            
        definition = "\n".join(paragraphs)
        definition = re.sub(r'\[\d+\]', '', definition)  # Remove citations
        
        return {
            "type": "definition",
            "concept": concept,
            "text": definition,
            "url": url,
            "timestamp": int(time.time())
        }

    def extract_stackexchange_qa(self, soup, url):
        """Extracts structured questions and answers from Q&A platforms."""
        title_el = soup.find("a", class_="question-hyperlink")
        if not title_el:
            title_el = soup.find("h1")
        if not title_el:
            return None
        question = title_el.get_text().strip()

        answer_div = soup.find("div", class_="accepted-answer")
        if not answer_div:
            answer_div = soup.find("div", class_="answer")
            
        if not answer_div:
            return None
            
        answer_body = answer_div.find("div", class_="js-post-body")
        if not answer_body:
            return None
            
        answer_text = answer_body.get_text(separator=" ").strip()
        lines = (line.strip() for line in answer_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        answer_clean = "\n".join(chunk for chunk in chunks if chunk)

        if len(answer_clean) < 100:
            return None

        return {
            "type": "qa",
            "question": question,
            "text": answer_clean,
            "url": url,
            "timestamp": int(time.time())
        }

    def extract_general_knowledge(self, soup, url):
        """Extracts Q&A and definitions dynamically from ANY general web page."""
        title_el = soup.find("title")
        page_title = title_el.get_text().strip() if title_el else ""
        
        extracted_units = []

        # 1. Look for Question-Answer structures (headings followed by text)
        headings = soup.find_all(["h1", "h2", "h3"])
        for h in headings:
            heading_text = h.get_text().strip()
            
            # Check if heading is a question (ends with '?' or contains question words)
            is_question = heading_text.endswith("?") or QUESTION_PATTERN.search(heading_text)
            if is_question and len(heading_text) > 10 and len(heading_text) < 150:
                # Find paragraphs immediately following this heading
                paragraphs = []
                sibling = h.find_next_sibling()
                while sibling and sibling.name not in ["h1", "h2", "h3"]:
                    if sibling.name == "p":
                        p_text = sibling.get_text().strip()
                        if len(p_text) > 40:
                            paragraphs.append(p_text)
                    sibling = sibling.find_next_sibling()
                    if len(paragraphs) >= 3:
                        break
                
                if paragraphs:
                    answer_text = "\n".join(paragraphs)
                    extracted_units.append({
                        "type": "qa",
                        "question": heading_text,
                        "text": answer_text,
                        "url": url,
                        "timestamp": int(time.time())
                    })

        # 2. If no Q&A headings found, fall back to extracting the lead text as a concept definition
        if not extracted_units and len(page_title) > 5:
            # Clean page title (e.g. remove " - Wikipedia", " | Spiegel" etc.)
            clean_title = re.split(r'\s+[-|•]\s+', page_title)[0].strip()
            
            paragraphs = []
            for p in soup.find_all("p"):
                p_text = p.get_text().strip()
                if len(p_text) > 60:
                    paragraphs.append(p_text)
                if len(paragraphs) >= 2:
                    break
            
            if paragraphs:
                definition_text = "\n".join(paragraphs)
                extracted_units.append({
                    "type": "definition",
                    "concept": clean_title,
                    "text": definition_text,
                    "url": url,
                    "timestamp": int(time.time())
                })

        return extracted_units

    def extract_links(self, html_content, base_url, target_domain):
        """Extracts internal and external links to explore the web randomly."""
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            if parsed.scheme in ("http", "https"):
                # Clean URL (discard anchors/queries to avoid loops)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)
        return links

    async def fetch_page(self, session, url):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        try:
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            pass
        return None

    async def save_to_jsonl(self, data):
        async with self.write_lock:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def worker(self, session):
        while self.pages_scraped < self.max_pages:
            target_domain = None
            for domain, queue in self.domain_queues.items():
                if not queue.empty() and domain not in self.active_domains:
                    last_time = self.last_fetch_time.get(domain, 0)
                    if time.time() - last_time >= self.delay_per_domain:
                        target_domain = domain
                        break
            
            if not target_domain:
                await asyncio.sleep(0.1)
                continue
                
            self.active_domains.add(target_domain)
            queue = self.domain_queues[target_domain]
            visited = self.domain_visited[target_domain]
            
            try:
                url = queue.get_nowait()
            except asyncio.QueueEmpty:
                self.active_domains.remove(target_domain)
                continue

            if url in visited:
                queue.task_done()
                self.active_domains.remove(target_domain)
                continue
                
            visited.add(url)
            
            self.last_fetch_time[target_domain] = time.time()
            html = await self.fetch_page(session, url)
            
            if html:
                soup = BeautifulSoup(html, "html.parser")
                
                # Check page type and extract knowledge structure accordingly
                extracted_data_list = []
                if "wikipedia.org" in target_domain:
                    wiki_def = self.extract_wikipedia_definition(soup, url)
                    if wiki_def:
                        extracted_data_list.append(wiki_def)
                elif "stackexchange.com" in target_domain or "stackoverflow.com" in target_domain:
                    se_qa = self.extract_stackexchange_qa(soup, url)
                    if se_qa:
                        extracted_data_list.append(se_qa)
                else:
                    # General web page parser (looks for Q&A structures and concept definitions)
                    extracted_data_list = self.extract_general_knowledge(soup, url)
                
                # Save extracted units
                for data in extracted_data_list:
                    self.pages_scraped += 1
                    await self.save_to_jsonl(data)
                    
                    if self.pages_scraped % 100 == 0:
                        logger.info(f"Progress: {self.pages_scraped} general knowledge units scraped.")

                # Extract and enqueue links (internal to stay on site, external to expand seeds)
                links = self.extract_links(html, url, target_domain)
                for link in links:
                    parsed_link = urlparse(link)
                    domain = parsed_link.netloc
                    
                    if domain:
                        # Dynamically add new domains to crawl to keep things random and general!
                        if domain not in self.domain_queues:
                            self.domain_queues[domain] = asyncio.Queue()
                            self.domain_visited[domain] = set()
                        
                        if link not in self.domain_visited[domain]:
                            self.domain_queues[domain].put_nowait(link)
                            
            queue.task_done()
            self.active_domains.remove(target_domain)
            await asyncio.sleep(0.01)

    async def run(self):
        logger.info(f"Starting General Knowledge FinnyCrawlerV2. Target: {self.max_pages} units.")
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(self.max_concurrency)]
            await asyncio.gather(*workers)
            
        logger.info(f"Finished. Scraped {self.pages_scraped} knowledge units to {self.output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinnyCrawler V2 - General Knowledge & Reasoning Extractor")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum units to scrape")
    parser.add_argument("--concurrency", type=int, default=40, help="Total parallel workers")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay per domain")
    parser.add_argument("--output", type=str, default="data/knowledge_data.jsonl", help="Output path")
    args = parser.parse_args()

    # Highly diverse seeds to start crawling the general web randomly
    seeds = [
        "https://en.wikipedia.org/wiki/Special:Random",  # Start randomly on English Wikipedia
        "https://de.wikipedia.org/wiki/Spezial:Zuf%C3%A4llige_Seite",  # Start randomly on German Wikipedia
        "https://news.ycombinator.com/",
        "https://www.bbc.com/news",
        "https://www.spiegel.de/",
        "https://www.gutenberg.org/",
        "https://www.nytimes.com/",
        "https://www.nature.com/",
        "https://www.britannica.com/",
        "https://archive.org/"
    ]

    crawler = FinnyKnowledgeCrawler(
        seed_urls=seeds,
        output_file=args.output,
        max_pages=args.limit,
        max_concurrency=args.concurrency,
        delay_per_domain=args.delay
    )

    asyncio.run(crawler.run())
