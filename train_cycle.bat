for /L %%i in (1,1,11) do (
    echo Train = %%i
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/clean.py
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/model_convert.py best
    @REM for /L %%j in (1,1,10) do (
    @REM echo SelfPlay = %%j
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/build/Release/AlphaPuyo.exe" "self"
    @REM )
    @REM for /L %%j in (1,1,5) do (
    @REM echo Eval = %%j
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/build/Release/AlphaPuyo.exe" "eval"
    @REM )
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/train_network.py
)
