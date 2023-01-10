from pathlib import Path
# from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
import sys
# import numpy as np
import shutil
import time
import multiprocessing as mp
# from itertools import product
import exchange_calendars as xcals
import warnings
warnings.simplefilter("ignore", UserWarning)

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402
from stock import make_stock  # noqa: E402
from candlestick import make_candlestick  # noqa: E402
sys.path.pop()

sys.path.append(str(ROOT / 'Yolo' / 'Code'))
from detect import detect_light, attempt_load, select_device  # noqa: E402

device = '0'
device = select_device(device)
weights = '/home/ubuntu/2022_VAIV_JSPARK/YOLOv7/yolov7/runs/train/yolov7-Kospi50P_5_3/weights/best.pt'
model = attempt_load(weights, map_location=device)


def default_vaiv() -> VAIV:
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kospi',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [1800, 650],
        'candle': 245,
        'linespace': 1,
        'candlewidth': 0.8,
        'folder': 'yolo',
        'name': 'Kospi50P',
        'style': 'default'  # default는 'classic'
    }
    vaiv.set_kwargs(**kwargs)

    return vaiv


def default_opt(vaiv: VAIV):
    opt = {
        'weights': '/home/ubuntu/2022_VAIV_JSPARK/YOLOv7/yolov7/runs/train/yolov7-Kospi50P_5_3/weights/best.pt',
        'conf_thres': 0.6,
        'device': '0',
        'model': model,
        'imgsz': 640,
        'iou_thres': 0.45,
        'trace': False,
        'vaiv': vaiv
    }
    return opt


def copy_image(tickers, trade_date, market):
    vaiv = default_vaiv()
    vaiv.set_kwargs(market=market)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    source = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/today/')
    source.mkdir(parents=True, exist_ok=True)

    notFound = {}
    for ticker in (tickers):
        vaiv.set_fname('png', ticker=ticker, trade_date=trade_date)
        vaiv.set_path(vaiv.common.image.get('images'))
        img = str(vaiv.path)
        try:
            shutil.copy(img, str(source / vaiv.path.name))
        except FileNotFoundError:
            print(img)
            notFound.update({ticker:['FileNotFoundError', 0, '', '']})
        except:
            print('Another Error: ', img)
            notFound.update({ticker:['FileNotFoundError', 0, '', '']})
    return notFound


def detect_list(tickers, trade_date, market='Kospi'):
    vaiv = default_vaiv()
    vaiv.set_kwargs(market=market)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    opt = default_opt(vaiv)
    opt['weights'] = '/home/ubuntu/2022_VAIV_JSPARK/YOLOv7/yolov7/runs/train/yolov7-Kospi50P_5_3/weights/best.pt'
    sell_tickers = {}
    source = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/today/')
    save_dir = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/predict/')
    source.mkdir(parents=True, exist_ok=True)

    notFound = {}
    files = []
    for ticker in (tickers):
        vaiv.set_fname('png', ticker=ticker, trade_date=trade_date)
        vaiv.set_path(vaiv.common.image.get('images'))
        img = str(vaiv.path)

        if vaiv.path.exists():
            files.append(img)
        else:
            notFound.update({ticker: ['FileNotFoundError', 0, '', '']})
        # try:
        #     shutil.copy(img, str(source / vaiv.path.name))
        # except FileNotFoundError:
        #     print(img)
        #     notFound.update({ticker:['FileNotFoundError', 0, '', '']})
        # except:
        #     print('Another Error: ', img)
        #     notFound.update({ticker:['FileNotFoundError', 0, '', '']})
        #     continue

    # if len(tickers) == len(notFound):
    #     return notFound
    # df = detect_light(**opt, source=source, save_dir=save_dir)

    if not files:
        return notFound

    df = detect_light(**opt, files=files)
    # df = df[df.Signal == 'sell']
    tickers = df.Ticker.tolist()
    probs = df.Probability.tolist()
    signals = df.Signal.tolist()
    starts = df.Start.tolist()
    ends = df.End.tolist()
    ret = {t: [s, p, start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
    ret.update(notFound)
    try:
        shutil.rmtree('/home/ubuntu/2022_VAIV_Dataset/flask/static/today')
    except FileNotFoundError:
        pass
    return ret


def detect_all():
    df = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/flask/static/Stock.csv', index_col=0)
    kospiTickers = df[df['Market'] == 'STK'].index.tolist()
    kosdaqTickers = df[df['Market'] == 'KSQ'].index.tolist()

    today = datetime.today()
    yesterday = today - timedelta(1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    XKRX = xcals.get_calendar("XKRX")
    trade_date = XKRX.next_session(yesterday).strftime('%Y-%m-%d')

    # p = Process(detect_list, args=(kospiTickers, trade_date, 'Kospi', ))
    Detection = dict()
    start = time.time()
    kospiDict = detect_first(kospiTickers, trade_date, 'Kospi')
    # print('Kospi_Dict: ', kospiDict)
    kospiT = time.time()
    kosdaqDict = detect_first(kosdaqTickers, trade_date, 'Kosdaq')
    kosdaqT = time.time()

    Detection.update(kospiDict)
    Detection.update(kosdaqDict)

    stock_list = list()
    for ticker, value in Detection.items():
        stock = pd.DataFrame({
            'Ticker': [ticker],
            'FullCode': [df.FullCode.loc[ticker]],
            'Symbol': [df.Symbol.loc[ticker]],
            'Signal': [value[0]],
            'Probability': [value[1]],
            'Start': [value[2]],
            'End': [value[3]],
        })
        stock.set_index('Ticker', drop=True, inplace=True)
        stock_list.append(stock)
    end = time.time()
    print('Kospi Time: ', kospiT - start)
    print('Kosdaq Time: ', kosdaqT - kospiT)
    print('After Detect Time: ', end - kosdaqT)
    detect = pd.concat(stock_list)
    detect.to_csv('./static/Detection.csv')


def detect_MarketFiles(trade_date, market):
    vaiv = default_vaiv()
    vaiv.set_kwargs(market=market)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    opt = default_opt(vaiv)
    filesPath = vaiv.common.image.get('images')
    # fileDetectStart = time.time()
    files = list(map(str, filesPath.glob(f'*{trade_date}.png')))
    # print('FileTime: ', time.time() - fileDetectStart)
    opt['weights'] = '/home/ubuntu/2022_VAIV_JSPARK/YOLOv7/yolov7/runs/train/yolov7-Kospi50P_5_3/weights/best.pt'
    df = detect_light(**opt, files=files)
    tickers = df.Ticker.tolist()
    probs = df.Probability.tolist()
    signals = df.Signal.tolist()
    starts = df.Start.tolist()
    ends = df.End.tolist()
    ret = {t: [s, p, start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
    return ret


def detectAllFiles():
    df = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/flask/static/Stock.csv', index_col=0)

    today = datetime.today()
    yesterday = today - timedelta(1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    XKRX = xcals.get_calendar("XKRX")
    trade_date = XKRX.next_session(yesterday).strftime('%Y-%m-%d')

    Detection = dict()
    start = time.time()
    kospiDict = detect_MarketFiles(trade_date, 'Kospi')
    # print('Kospi_Dict: ', kospiDict)
    kospiT = time.time()
    kosdaqDict = detect_MarketFiles(trade_date, 'Kosdaq')
    kosdaqT = time.time()
    Detection.update(kospiDict)
    Detection.update(kosdaqDict)

    stock_list = list()
    for ticker, value in Detection.items():
        stock = pd.DataFrame({
            'Ticker': [ticker],
            'FullCode': [df.FullCode.loc[ticker]],
            'Symbol': [df.Symbol.loc[ticker]],
            'Signal': [value[0]],
            'Probability': [value[1]],
            'Start': [value[2]],
            'End': [value[3]],
        })
        stock.set_index('Ticker', drop=True, inplace=True)
        stock_list.append(stock)
    end = time.time()
    print('Kospi Time: ', kospiT - start)
    print('Kosdaq Time: ', kosdaqT - kospiT)
    print('After Detect Time: ', end - kosdaqT)
    detect = pd.concat(stock_list)
    detect.to_csv('./static/Detection.csv')


def make_process(vaiv, ticker, trade_date, result_dict):
    vaiv.set_kwargs(ticker=ticker)
    vaiv.set_kwargs(trade_date=trade_date)
    stock = make_stock(vaiv, end=trade_date, save=False)

    condition1 = len(stock) > vaiv.kwargs.get('candle')
    condition2 = trade_date in stock.index
    if condition1 & condition2:
        start = stock.index[-245]
        pred = pd.Series({'Start': start, 'End': trade_date, 'Date': trade_date})
        result_dict['price'][ticker] = stock.loc[trade_date, 'Close']
        make_candlestick(vaiv, stock, pred)
        result_dict['files'].append(str(vaiv.common.image.get('images') / f'{ticker}_{trade_date}.png'))
    else:
        result_dict['notFound'].update({ticker: ['FileNotFoundError', 0, 0, '', '']})


def detect_Test(tickers, trade_date, market):
    s1 = time.time()
    source = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/RealTime/')
    source.mkdir(parents=True, exist_ok=True)
    save_dir = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/predict/')
    vaiv = default_vaiv()
    vaiv.set_kwargs(market=market)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image(source)
    vaiv.make_dir(common=True, image=True)
    vaiv.set_labeling()
    opt = default_opt(vaiv)
    opt['weights'] = '/home/ubuntu/2022_VAIV_JSPARK/YOLOv7/yolov7/runs/train/yolov7-Kospi50P_5_3/weights/best.pt'

    manager = mp.Manager()
    result_dict = manager.dict()
    result_dict['notFound'] = manager.dict()
    result_dict['price'] = manager.dict()
    result_dict['files'] = manager.list()
    e1 = time.time()
    readyT = e1-s1

    jobs = []
    s2 = time.time()
    for ticker in tickers:
        p = mp.Process(target=make_process, args=(vaiv, ticker, trade_date, result_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    e2 = time.time()
    makeT = e2 - s2

    s4 = time.time()
    if len(tickers) == len(result_dict['notFound']):
        return result_dict['notFound']
    df = detect_light(**opt, source=vaiv.common.image.get('images'), save_dir=save_dir, files=result_dict['files'])
    e4 = time.time()

    detectT = e4-s4

    s5 = time.time()
    tickers = df.Ticker.tolist()
    probs = df.Probability.tolist()
    signals = df.Signal.tolist()
    starts = df.Start.tolist()
    ends = df.End.tolist()
    ret = {t: [s, p, int(result_dict['price'][t]), start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
    ret.update(result_dict['notFound'])

    e5 = time.time()
    returnT = e5 - s5

    print('Times: ', [readyT, makeT, detectT, returnT])
    return ret


def detect_first(tickers, trade_date, market):
    s1 = time.time()
    source = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/RealTime/')
    source.mkdir(parents=True, exist_ok=True)
    save_dir = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/predict/')
    vaiv = default_vaiv()
    vaiv.set_kwargs(market=market)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image(source)
    vaiv.make_dir(common=True, image=True)
    vaiv.set_labeling()
    opt = default_opt(vaiv)
    opt['weights'] = '/home/ubuntu/2022_VAIV_JSPARK/YOLOv7/yolov7/runs/train/yolov7-Kospi50P_5_3/weights/best.pt'

    notFound = {}
    e1 = time.time()
    # print(f'Ready: {round(e1-s1, 4)}s')
    readyT = e1-s1

    stock_t = 0
    candle_t = 0
    ticker_count = len(tickers)
    price = {}
    files = []

    jobs = []
    for ticker in tickers:
        s2 = time.time()
        vaiv.set_kwargs(ticker=ticker)
        vaiv.set_kwargs(trade_date=trade_date)
        stock = make_stock(vaiv, end=trade_date, save=False)
        e2 = time.time()
        stock_t += e2 - s2

        s3 = time.time()
        condition1 = len(stock) > vaiv.kwargs.get('candle')
        condition2 = trade_date in stock.index
        # if condition1 & condition2:
        if condition1:
            start = stock.index[-245]
            trade_date = stock.index[-1]  # 임시
            vaiv.set_kwargs(trade_date=trade_date)  # 임시
            pred = pd.Series({'Start': start, 'End': trade_date, 'Date': trade_date})
            price[ticker] = stock.loc[trade_date, 'Close']
            # make_candlestick(vaiv, stock, pred)
            p = mp.Process(target=make_candlestick, args=(vaiv, stock, pred, ))
            jobs.append(p)
            p.start()
            files.append(str(vaiv.common.image.get('images') / f'{ticker}_{trade_date}.png'))
        else:
            notFound.update({ticker: ['FileNotFoundError', 0, 0, '', '']})
            ticker_count -= 1

        e3 = time.time()
        candle_t += e3 - s3
    for proc in jobs:
        proc.join()
    # print(f'{ticker_count}/{len(tickers)} Total Stock: {round(stock_t, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Total Candlestick: {round(candle_t, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Avg Stock: {round(stock_t / ticker_count, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Avg Candlestick: {round(candle_t / ticker_count, 2)}s')

    s4 = time.time()
    if len(tickers) == len(notFound):
        return notFound
        # return notFound, [0, 0, 0, 0, 0]
    df = detect_light(**opt, source=vaiv.common.image.get('images'), save_dir=save_dir, files=files)
    e4 = time.time()
    # print(f'Tickers Detect: {round(e4-s4, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Tickers Detect: {round(e4-s4, 2)}s')
    detectT = e4-s4

    s5 = time.time()
    tickers = df.Ticker.tolist()
    probs = df.Probability.tolist()
    signals = df.Signal.tolist()
    starts = df.Start.tolist()
    ends = df.End.tolist()
    ret = {t: [s, p, int(price[t]), start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
    ret.update(notFound)

    # try:
    #     shutil.rmtree(str(source))
    # except FileNotFoundError:
    #     pass

    e5 = time.time()
    # print(f'{len(tickers)} Tickers Return: {round(e5 - s5, 4)}s')
    returnT = e5 - s5
    print('Times: ', [readyT, stock_t, candle_t, detectT, returnT])
    return ret
    # return ret, [readyT, stock_t, candle_t, detectT, returnT]


if __name__ == '__main__':
    # detectAllFiles()
    # detect_all()

    vaiv = default_vaiv()
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()
    tickers = df.Ticker.tolist()[:40]

    s0 = time.time()
    # tickers = ['005930', '000020', '095570', '006840', '039570']
    # tickers = ['006390']
    ret = detect_Test(tickers=tickers, trade_date='2022-12-13', market='Kospi')
    e0 = time.time()
    print(ret)
    print(f'{len(tickers)} Tickers Total: {round(e0-s0, 2)}s')

    # totalT = 0
    # AvgTimes = [0, 0, 0, 0, 0]
    # notFound = 0
    # for ticker in tickers:
    #     s0 = time.time()
    #     ret, times = detect_first(tickers=[ticker], trade_date='2022-12-13', market='Kospi')
    #     e0 = time.time()
    #     if ret[ticker][0] == 'FileNotFoundError':
    #         notFound += 1
    #         continue
    #     AvgTimes = [AvgTimes[i] + times[i] for i in range(5)]
    #     totalT += e0 - s0
    # AvgTimes = [t / (len(tickers) - notFound) for t in AvgTimes]
    # totalT /= (len(tickers) - notFound)
    # print('--------------------------------------------------------------------------')
    # print(f'Ready: {round(AvgTimes[0], 4)}s')
    # print(f'{len(tickers) - notFound} Avg Stock: {round(AvgTimes[1], 2)}s')
    # print(f'{len(tickers) - notFound} Avg Candlestick: {round(AvgTimes[2], 2)}s')
    # print(f'{len(tickers) - notFound} Tickers Detect: {round(AvgTimes[3], 2)}s')
    # print(f'{len(tickers) - notFound} Tickers Return: {round(AvgTimes[4], 4)}s')
    # print(f'{len(tickers) - notFound} Tickers Total: {round(totalT, 2)}s')

    # sell_tickers = detect_list(tickers)
    # print(len(sell_tickers))
    # print(sell_tickers)
