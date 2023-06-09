# open-campus-object-detection


### Instructions - Setup & Run

Step 1: Install required python packages (see requirements.txt)

Step 2: Setup env.py file as shown in env.py.example

Step 3: Run the application as follows

Terminal/Command line input:
```properties
python main.py
```

### Current State - Open

*Felix*
- Fill database with texts for all labels
- Regain api key

*Tobias*
- Fit text into bounding boxes (approx.) or think of another solution.
- Maybe try to limit predictions to the top 2 (if possible) 


### Info

Use the api.get_response method only like this 

```
api.get_label_response(label, test=True)
```

The database will be filled for all YoloV5 Lables to be able to display it at Open Campus

### Structure - UML Diagram

![Structure - uml-diagram.svg](assets/uml-diagram.svg)

``<uses>`` indicates a dependency injection, e.g.: ``A <uses> B`` means ``B is injected into A``