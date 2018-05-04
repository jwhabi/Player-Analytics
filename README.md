# Player-Analytics
Prediction Player potential and Position based on FIFA player data and Match recordings


Note: Download the 'implementation_package.rar' file to get a executable package of the analysis.

**To run: 
Extract the rar file, and start a local server through command line by traversing to the newly created directory in command line 
and executing 'python filechooser.py'. 
Then open the link 127.0.0.1:5000/Test on your browser. **

Sports Analytics usually involves identifying and collecting massive amounts of data for a set of leagues, teams and/or
players in a sport to extract meaningful insights which might not be prominent by just observation. This collection of
relevant and historical statistics as well as player and team attributes, when properly applied, can be used to provide a
competitive advantage to a team or an individual player. Analyzing a collection of datasets with information like the one
mentioned above can lead to two kinds of analysis: on-field analysis and off-field analysis. Off -field analysis deals with
the business side of the sport like current market value, monetary compensations, betting odds, winning likelihoods, etc.
while the former deals with generating insights from data like optimal team tactics, player improvement and training
insights, success factors, player readiness, talent analysis, etc. The current problem statement focuses on On-field
analytics for players and teams in European football (soccer) in particular.

**Problem Statement**

The goal of this project will be to predict, given data of an individual, the potential of the said individual and the ideal
position of play for the individual. The output of this analysis can be used by any individual to introspect their likelihood
of being a professional soccer player. Moreover, the same analysis can be used by professional teams and coaches to see
the potential of a player as well as which position is the player most in line with, based on current available information,
and whether they should be brought into the team. The potential in question, will be a rating measure of a scale of 1-100.
It indicates a prospective future 'overall' rating a player can reach based on their current attributes. Positions are indicative
of the position of a player on the field, i.e. 'Forward', 'Midfielder', 'Defender' and 'Goalkeeper'.
The two desired predictions will be implemented via a comparative study of modelling technique implementations. Player
potential will be implemented with techniques of Regression and Decision Tree classifiers whereas Player position will be
implemented with techniques of Bayesian Classifiers and Clustering. For each of the two areas of potential and position,
the most efficient model of the two discussed will be used for implementation.

**Data Description**

The dataset is available on Kaggle (https://www.kaggle.com/doctorclo/can-you-be-a-good-football-player/data ). The
dataset was originally available in the SQLite format. We converted it into csv files for easier processing using python. 8
datasets were available, but we used only three for our project -
**Player Data**
We had data for 11,060 players. This data gives us the physical attributes of the players. This data also gives us unique
player ids for each player.
It contained id, fifa_player_id which were unique for every player. Also, it had the birthday, height and weight for every
player.
**Player Attributes Data**
We had data for 183978 records of player attributes. These attributes describe a player’s proficiency in different tasks.
This data has multiple instances for each player, as data is updated for each player multiple times during a season. It
contained multiple records for each player which were recorded over the years. It ranked players from 0-100 in various
categories. The data had 43 columns most of which are self-explanatory, a few of which have been listed below. The
complete details can be found on the Kaggle link. All of them have not been explained here due to space constraints.
**Match Data**
We had data for 25980 matches played in different leagues. This data gives us the teams which played the match. The
players that played for the team. And their position coordinates on the field.
It contained the ids of all the players that played in every match along with their position coordinates on the field, which
helped us in identifying there positions.
The original datasets were compiled using these datasources:
• http://football-data.mx-api.enetscores.com/ : scores, lineup, team formation and events
• http://www.football-data.co.uk/
• http://sofifa.com/ : players and teams attributes from EA Sports FIFA games. FIFA series and all FIFA assets property of EA
Sports.

**The Data Processing folder contains files for data processing and feature extraction
The Data Modeling folder consists of the discussed 4 modelling implementations
The implementation folder consists of the source code for the flask app created
The implementation rar file contains the actual distribution package for running this predictive report. **
** For further details, the entire project report can be found on this link:
https://drive.google.com/file/d/1s6hOQy2qrnAsZQEjghq68XPnPs2xUay1/view?usp=sharing **
