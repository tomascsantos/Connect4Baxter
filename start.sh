#!/bin/bash

cd $HOME/ros_workspaces/final_proj/

mate-terminal --tab --title=actionServer -e "./baxter.sh ada.local"
mate-terminal --tab --title=moveIt -e "./baxter.sh ada.local"
mate-terminal --tab --title=alvar -e "./baxter.sh ada.local"
mate-terminal --tab --title=dev -e "./baxter.sh ada.local"
mate-terminal --tab --title=test -e "./baxter.sh ada.local"

firefox https://ucb-ee106.github.io/ee106a-fa19/
firefox https://docs.google.com/document/d/1ZcTH0kQdSqJmBHe5Ws9JtNcbLNLQGY5Q27oMFfkAhKE/edit

xdg-open ./.resources/lab3.pdf
xdg-open ./.resources/lab4.pdf
xdg-open ./.resources/lab5.pdf
xdg-open ./.resources/lab6.pdf
xdg-open ./.resources/lab7.pdf
xdg-open ./.resources/lab8.pdf

