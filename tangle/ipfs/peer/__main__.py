import asyncio

from .main import main

if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(loop))
        loop.close()
    except KeyboardInterrupt:
        pass
