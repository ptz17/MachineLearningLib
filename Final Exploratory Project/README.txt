To run the script for the final exploratory project, please follow these steps in the CADE linux terminal.

1) Clone repository onto local computer running: "git clone https://github.com/ptz17/MachineLearningLib.git"
2) Change directories into "MachineLearningLib/Final Exploratory Project"
3) Change the permissions for the run.sh shell script by running: "chmod u+x run.sh"
4) Execute the shell script using: "./run.sh"

The program may print out a lot of data to the terminal that is useful for viewing updates to the learning parameters
and losses throughout execution for people who are familiar with the code. They may be hard to interpret for new comers,
however, at the end of execution, loss curves should be displayed for each script. I don't believe the next scipt will
begin executing until the plot from the previous script has been exited. 

These script assumes that torch version 1.9.1 is already installed on the CADE computers.