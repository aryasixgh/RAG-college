import requests
from bs4 import BeautifulSoup
import os
import re
import time

course_code = 6853
cookie_data = "ts5h08k0thqpjimbdm3tcvgi87"
BASE_URL = "https://mydy.dypatil.edu"
START_PAGE = f"https://mydy.dypatil.edu/rait/course/view.php?id={course_code}"
cookies = {
    "MoodleSession": f"{cookie_data}"
}
#subject_name = input("Enter subject :")
OUTPUT_DIR = r"C:\Users\aryas\OneDrive\Documents\College\Sem 6\dma\Downloaded_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()

def get_flexpaper_links(course_url):
    print(f"Fetching course page: {course_url}")
    resp = session.get(course_url, cookies=cookies)
    if not resp.ok:
        print(f"Failed to load course page {course_url}: {resp.status_code}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r'/mod/flexpaper|url|presentation/view\.php\?id=\d+', href):
            if not href.startswith("http"):
                href = BASE_URL + href

            #finding span text
            span = a.find("span", class_="instancename")
            span_text = ""
            if span:
                # Remove nested accesshide span (usually like "Presentation (Secured PDF)")
                accesshide = span.find("span", class_="accesshide")
                if accesshide:
                    accesshide.extract()  # Remove it from the tree

                span_text = span.get_text(strip=True)  # Clean visible span text

            links.append([href, span_text])
    print(f"Found {len(links)} flexpaper viewer links") 
    return links

def extract_pdf_url(flexpaper_url, span_text):
    print(f"\n🔄 Visiting: {flexpaper_url}")

    #session.get(flexpaper_url, cookies=cookies)
    # time.sleep(1)

    resp = session.get(flexpaper_url, cookies=cookies)
    # time.sleep(1)

    if not resp.ok:
        print(f"❌ Failed to load {flexpaper_url}: {resp.status_code}")
        return None, None

    html = resp.text


    match = re.search(r"PDFFile\s*:\s*'([^']+\.pdf)'", html) or re.search(r'<a\s+[^>]*href=["\']([^"\']+\.pdf)["\']', html, re.IGNORECASE)
    if match:
        pdf_url = match.group(1)
        filename = pdf_url.split('/')[-1]

        output_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(output_path):
            print(f"⏭️ Skipping (already downloaded): {filename}")
            return None, None

        print(f"✅ Found PDF to download: {filename} \n🔗 URL: {pdf_url} \n📋 Name: {span_text}")
        return filename, pdf_url
    else:
        print(f"❌ No PDF URL found on page \n📋 Name: {span_text}")
        return None, None
    
def download_pdf(pdf_url, filename):
    print(f"⬇️ Downloading: {filename}")
    resp = session.get(pdf_url, cookies=cookies)
    if resp.ok:
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"✅ Saved: {filename}")
    else:
        print(f"❌ Failed to download {pdf_url}: {resp.status_code}")

def main():
    flexpaper_links = get_flexpaper_links(START_PAGE)
    pdfCount=0
    for flex_url, span_text in flexpaper_links:
        filename, pdf_url = extract_pdf_url(flex_url, span_text)
        if pdf_url:
            pdfCount+=1
            print(pdfCount)
        if filename and pdf_url:
            download_pdf(pdf_url, filename)

if __name__ == "__main__":
    main()
