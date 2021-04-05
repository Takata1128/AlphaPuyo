for /L %%i in (1,1,30) do (
    echo Train = %%i
    "**/python.exe" ./python/clean.py
    "**/python.exe" ./python/model_convert.py best
    "./build/Release/AlphaPuyo.exe" "self"
    "./build/Release/AlphaPuyo.exe" "eval"
    "**/python.exe" ./python/train_network.py
)
