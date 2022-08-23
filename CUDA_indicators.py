

### Dedicated to writing code to compute EMA SMA etc on GPU
 ## https://github.com/AshwinAgrawal16/Technical-Indicators-In-C-     Has some preimplemented
from numba import vectorize, cuda
import numpy as np
import time

def to_device(OHLCV):
    """
    Pass data to gpu and create arrays to hold results
    OHLCV - Open High Low Close Volume"""
    #Send OHLVC to GPU
    OHLCV=OHLCV.transpose().astype('float32')
    OHLCV=cuda.to_device(OHLCV)
    [O,H,L,C,V] = OHLCV
    #Get array length
    n = len(O)
    #Create output buffers
    EMA_short = cuda.device_array(shape=(n,), dtype=np.float32)
    EMA_long = cuda.device_array(shape=(n,), dtype=np.float32)
    EMA_p1 = cuda.device_array(shape=(n,), dtype=np.float32)
    EMA_p2 = cuda.device_array(shape=(n,), dtype=np.float32)
    MACD = cuda.device_array(shape=(n,), dtype=np.float32)
    RSI = cuda.device_array(shape=(n,), dtype=np.float32)
    WilliamR = cuda.device_array(shape=(n,), dtype=np.float32)
    return O,H,L,C,V, MACD,RSI,WilliamR, EMA_long, EMA_short,EMA_p1,EMA_p2

def to_host(identifiers):
    [MACD,RSI,WilliamR] = identifiers
    for i in [MACD,RSI,WilliamR]:
        i = i.copy_to_host()
    return [MACD,RSI,WilliamR]

@vectorize(['float32(float32, float32)'], target='cuda')
def macd_ufunc(shortema,longema):
    return shortema-longema

@vectorize(['float32(float32, float32,float32,float32,int32,int32)'], target='cuda')
def ema_ufunc(close,EMA,EMA_p1,EMA_p2,period,smoothing=2):
    EMA_p1 = close*(smoothing/(1+period))
    EMA_p2 = close*(1-(smoothing/(1+period)))
    return EMA_p1#,EMA_p2

if __name__=='__main__':
    OHLVC=np.arange(0,5*7,1).reshape(7,5)
    O,H,L,C,V, MACD,RSI,WilliamR, EMA_long, EMA_short,EMA_p1,EMA_p2=to_device(OHLVC)
    period=2
    EMA_p1,EMA_p2=ema_ufunc(C,EMA_short,EMA_p1,EMA_p1,period=[period],smoothing=[2])
    EMA = EMA_p1[period::]+EMA_p2[:period:]