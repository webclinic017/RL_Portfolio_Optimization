# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as soup
'''
brew install phantomjs
or
brew tap homebrew/cask
brew cask install chromedriver
'''
from selenium.webdriver.chrome.webdriver import WebDriver


def get_etf_holdings(etf_symbol):
    '''
       etf_symbol: str

       return: list of stock symbols
       '''
    url = 'https://www.barchart.com/stocks/quotes/{}/constituents?page=all'.format(etf_symbol)

    # Loads the ETF constituents page and reads the holdings table
    browser = WebDriver()  # webdriver.PhantomJS()
    browser.get(url)
    html = browser.page_source
    page_soup = soup(html, "html")
    table = page_soup.find('div', {'class': 'bc-datatable'})

    # Reads the holdings table line by line and appends each asset to a
    # dictionary along with the holdings percentage
    asset_dict = {}
    for row in table.select('tr')[1:-1]:
        try:
            cells = row.select('td')
            # print(row)
            symbol = cells[0].get_text().strip()
            # print(symbol)
            name = cells[1].text.strip()
            celltext = cells[2].get_text().strip()
            percent = float(celltext.rstrip('%'))
            shares = int(cells[3].text.strip().replace(',', ''))
            if symbol != "" and percent != 0.0:
                asset_dict[symbol] = {
                    'name': name,
                    'percent': percent,
                    'shares': shares,
                }
        except BaseException as ex:
            print(ex)
    browser.quit()
    return list(asset_dict.keys())


if __name__ == "__main__":
    print(get_etf_holdings('XLE'))
