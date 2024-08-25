import requests
from bs4 import BeautifulSoup
import logging
import csv

def scrape_specific_content(url, target_class):
    documents = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Use a more general class selector or find by partial match
        elements = soup.find_all(class_=lambda c: c and target_class in c.split())
        for element in elements:
            content = element.get_text(strip=True)
            documents.append({'url': url, 'content': content})
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
    
    return documents

def save_to_csv(documents, filename):
    # Define the CSV header
    fieldnames = ['url', 'content']
    
    # Write the documents to a CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for document in documents:
            writer.writerow(document)

# Example usage:
scholarship_url = "https://lgu.edu.pk/scholarships/"
target_class = "fusion-layout-column"

# Scrape the specific content
documents = scrape_specific_content(scholarship_url, target_class)

# Save the scraped content to a CSV file
save_to_csv(documents, 'scholarships_content.csv')
