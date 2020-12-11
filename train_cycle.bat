for /L %%i in (1,1,10) do (
    echo Train = %%i
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/model_convert.py
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/build/Release/AlphaPuyo.exe"
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/train_network.py
    "C:/Users/rokahikou/.virtualenvs/AlphaPuyo-3tcf6cQj/Scripts/python.exe" C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/python/evaluate_network.py
)
