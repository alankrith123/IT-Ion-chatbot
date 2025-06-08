from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

driver.get("https://itionsolutions.com")
time.sleep(3)

text = driver.find_element("tag name", "body").text

with open("company_data.txt", "w", encoding="utf-8") as f:
    f.write(text)

driver.quit()
print("Scraped visible content and saved to company_data.txt")
