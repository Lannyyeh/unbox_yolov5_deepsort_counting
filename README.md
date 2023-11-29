# 碩一課程Cyber-Physics System 的專案

## 簡介

- 功能可能比較簡陋。
- 我們小組關於物聯網的大作業，是打算做一個攝像頭人物檢測的專案。
- 攝像頭那邊應該要是一個簡單的IOT設備，應該要是無法進行太複雜的運算，只能將獲取到的信息傳輸到計算中心進行處理，再接收一個結果，這樣簡單的設備；但是，由於我們沒有這種嵌入式設備，所以目前client和server用的都是普通pc，對於client來說是性能是冗餘的。
- 總之，傳輸這邊最後交由到另外兩位大佬負責，我負責的部分是server端的人物識別和追蹤。輸入會是根據rtmp協議獲取的圖像，code的功能只要從 `cv2.VideoCapture()` 開始就可以了。模型使用的是fork過來的原版，只有在具體功能上做一些修改，以及可能因為版本較舊造成的error進行一點點處理，改的不多，基本能直接跑通。

## 環境配置和流程

1. 首先需要裝CUDA和Cudnn，以及pytorch，我這邊最後使用的cuda 12.0和pytorch 11.8
2. 環境配置使用的是conda，可以直接使用[配置文件](./environment.yml)

    ``` bash
    conda env create -f environment.yml --name new_env_name
    ```

3. 原本作者實現的是檢測經典行人影片上行和下行統計

    ``` bash
    python main.py
    ```

4. 基於原本程式寫的檢測人物進入和離開一個區域的統計。
    - 測試資源目前還是原本的行人影片

    ``` bash
    python CountPeople.py
    ```

## 對原版的修改記錄

- 原版程式中使用了一些`np.float` 和 `np.int`此類的格式，根據error信息，這種寫法已被捨棄，可以直接使用`float` 和 `int`；或者用`np.float64` 和 `np.float32`這樣的寫法。
- 其餘的可以參照[作者原本的說明](./README_origin.md)
