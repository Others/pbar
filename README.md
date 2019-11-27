pbar: An Automatic Progress Bar
===============================

Instructions for Development
----------------------------
1) Start up the `vagrant` environment  
    ```shell script
    vagrant up; vagrant ssh; vagrant halt
    ```
    This can be very slow, so be patient
2) Install dependencies in the shell that's open now:
    ```shell script
    pip3 install -r requirements.txt
    ```
   (This may not work/may take some massaging)
3) Now you should be able to run the program
    ```shell script
    python3 pbar.py <LONG RUNNING COMMAND>
    ```