# LOL-Win-Prediction

### Abstract
The purpose of this project is to use the data taken after 10 minutes of solo ranked gameplay
and predict a win. There are 40 attributes recorded for each game: gameID, a blue team
win/loss, and the other 38 are split between each team for different goals achieved. These goals
include kills, deaths, towers destroyed, etc. This dataset has approximately 10,000 games saved.
After doing proper preprocessing and experimenting with this data, three models were used for
prediction: Logistic Regression, K-neighbors, and Support Vector Classification (SVC). All models
did exceptionally well but two proved about 5% more accurate. After running the models
several times, the following are average scores generated: Log-Reg = 0.73, KN = 0.68, SVC = 0.73.

### What is included? 
* Python code
* Report giving details on experiment
* [Dataset](https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
