from selenium import webdriver

driver = webdriver.Chrome('https://sites.google.com/chromium.org/driver')
driver.get('hi.html')
driver.maximize_window()
driver.quit()
