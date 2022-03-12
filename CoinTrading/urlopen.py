from time import time
from urllib.request import Request, urlopen

urls = [f"https://www.google.co.kr/search?q={query}" for query in ['apple', 'pear', 'grape', 'pineapple', 'orange', 'strawberry']]
begin = time()
result = []
for url in urls:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urlopen(request)
    page = response.read()
    result.append(len(page))

print(result)
end = time()
print(f"실행 시간: {end-begin}초")
