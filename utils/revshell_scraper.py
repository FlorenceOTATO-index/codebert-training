"""
Reverse Shell Scraper for revshells.com
---------------------------------------

This script uses Selenium WebDriver to automatically scrape reverse shell payloads
from the website https://www.revshells.com.

Workflow:
1. Launch a configured Chrome WebDriver (optionally headless).
2. Navigate to revshells.com and locate all shell type buttons on the left panel.
3. Click each button sequentially and extract the corresponding reverse shell code
   from the right-side display area.
4. Save all collected results (button text, full content, and code snippet) into
   a JSON file (`revshells_contents.json`).

Output: `revshells_contents.json` containing structured reverse shell payloads.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json

# Target website URL
BASE_URL = "https://www.revshells.com"


def init_driver():
    """Initialize WebDriver with browser options"""
    options = webdriver.ChromeOptions()
    # Optional: enable headless mode (browser window not shown)
    # options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")  # Large enough to display both panels
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )

    # Initialize driver
    driver = webdriver.Chrome(options=options)
    return driver


def get_shell_contents():
    """Click all buttons on the left panel and extract the content displayed on the right"""
    driver = init_driver()
    results = []

    try:
        # Access the target website
        driver.get(BASE_URL)
        print(f"Successfully loaded site: {BASE_URL}")

        # Wait for page to fully load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)  # Extra wait for JS execution

        # Locate left panel container (adjust selector if site structure changes)
        left_panel = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.shells-list"))
        )

        # Get all buttons inside left panel
        buttons = left_panel.find_elements(By.CSS_SELECTOR, "button")
        print(f"Found {len(buttons)} buttons in the left panel")

        # Iterate over each button
        for i, button in enumerate(buttons, 1):
            try:
                # Get button text (shell type name)
                button_text = button.text.strip()
                print(f"\nClicking button {i}: {button_text if button_text else f'Button {i}'}")

                # Ensure visibility and click
                driver.execute_script("arguments[0].scrollIntoView();", button)
                time.sleep(0.5)
                button.click()

                # Wait for right content area to update
                time.sleep(1)

                # Locate right content container
                right_content = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.shell-code"))
                )

                # Extract visible text
                content_text = right_content.text.strip()

                # Extract code block if present
                code_content = ""
                try:
                    code_block = right_content.find_element(By.CSS_SELECTOR, "pre")
                    code_content = code_block.text.strip()
                except NoSuchElementException:
                    pass

                # Save results
                results.append({
                    "button_index": i,
                    "button_text": button_text,
                    "full_content": content_text,
                    "code": code_content
                })

                print(f"Extracted content for button {i}")

            except TimeoutException:
                print(f"Timeout: content not loaded after clicking button {i}")
            except Exception as e:
                print(f"Error processing button {i}: {str(e)}")

    except Exception as e:
        print(f"Error accessing website: {str(e)}")
    finally:
        # Close browser
        driver.quit()
        print("\nBrowser closed")

    # Save results to JSON file
    with open('revshells_contents.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Completed: extracted {len(results)} items, saved to revshells_contents.json")

    return results


if __name__ == "__main__":
    get_shell_contents()
