# open-campus-object-detection


To run the application. Use: 

```properties
python main.py
```

There is an example env.py named env.py.example that shows how to setup this file.

*Current State - Open*

- Show ChatGPT logo?

Felix:
- Fill database with texts for all labels
- Regain api key

Tobias:
- Fit text into bounding boxes (approx.) or think of another solution. -> implemented
- Maybe try to limit predictions to the top 2 (if possible) 


*Info*

Use the api.get_response method only like this 

```
api.get_label_response(label, test=True)
```

The database will be filled for all YoloV5 Lables to be able to display it at Open Campus
