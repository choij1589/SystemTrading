import asyncio
 
async def hello():    # async def로 네이티브 코루틴을 만듦
    print('Hello, world!')

async def add(a, b):
    print(f"add: {a}+{b}")
    await asyncio.sleep(1.)
    return a+b

async def print_add(a, b):
    result = await add(a, b)
    print(f"print_add: {a}+{b} = {result}")

loop = asyncio.get_event_loop()     # 이벤트 루프를 얻음
loop.run_until_complete(print_add(1, 2))    # hello가 끝날 때까지 기다림
loop.close()                        # 이벤트 루프를 닫음
