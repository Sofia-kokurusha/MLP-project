# MLP-project

this is my project on the course "Machine Learning in Practice".

Topic of my project : "Motor Vehicle Collisions - Crashes" <br>

This data set explores the motor crashes in New York. 
Data is taken from NYC OpenData (https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)

Goal of my project: 
To predict the number of car crashes in the future based on the given location and day of the week

My project could be helpful for traffic analysis and further control by placing more traffic lights in dangerous areas. Or even placing police officers during particular days on specific areas.

Explanatory variables (x): 
location coordinates
day of the week

Target value (y): 
number of car crashes ( in a given day and the location)
classify per contributing factor


- for ecolyon to work: activate the env. in the terminal (cd, conda activate, poetry install)
- drop the rows that are empty 
- rescale the location range of the values. better to used latitude and longitude
- x: contributing factor, location, day of the week 
- this is a regression model, since I have a numerical prediction 
- no clustering, better to mention it in the further development 
- cat boost regressor for the model 


cd /path/to/the/notebook/
conda activate mlp-final-project
poetry install 

poetry add ..(jupyter, matplotlib, scikit-learn)
