for /L %%i in (1,1,10) do (
    echo Train = %%i
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/clean.py
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/model_convert.py
    for /L %%j in (1,1,10) do (
    echo SelfPlay = %%j
        "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/build/Release/AlphaPuyo.exe" "self"
    )
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/train_network.py
    for /L %%j in (1,1,5) do (
    echo Eval = %%j
        "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/build/Release/AlphaPuyo.exe" "eval"
    )
)
