import time
from kiui.timer import sync_timer

# `TIMER=1 python tests/test_timer.py`

with sync_timer("test"):
    time.sleep(1)

@sync_timer("test2")
def test2():
    time.sleep(1)

test2()