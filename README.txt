
Files:
Bay_LGBM.py : Baysian+LGBM ML model code
Traffic.py : Normal traffic generator code
Attack.py : Attack generator code
l3_learning_edit.py : POX controller edited code
Topology.mn : Mininet topology constructor
trained_model.sav : ML trained model
full.csv : Dataset
.PNG : Outputs obtained

Files locations:

Can create a directory anywhere in VM ( I created Mininet/Project directory) to add the following files inside:
Traffic.py
Attack.py
Topology.mn

Copy to /pox directory:
Bay_LGBM.py
full.csv
trained_model.sav (optional)

Copy to /pox/pox/forwarding directory:
l3_learning_edit.py


Procedure:

1)Open a terminal
Type the following:
>> cd mininet
>> sudo python examples/miniedit.py

-- Mininet will open------

2) In Mininet interface
File -> Open -> select ./mininet/Project/Topology.mn

-- Topology will appear in mininet

3) Open another terminal
Type the following
>> cd pox
>> ./pox.py forwarding.l3_learning_edit

-----pox is ready to work-------------

4) In Mininet, Click Run button

-----topology will begin to run------
-----POX controller shows the connected switches information---------

5) In Mininet, 
Right click on h1 hoat -> terminal
--------- h1 terminal will appear----------
	a) Type the following in h1 terminal
		>> cd Project
		>> sudo python Attack.py
		
	-----------Attack from h1 will begin------------
		
6) Simultaneously in Minnet,
Right click on any other hosts (I used h23) -> terminal
-----------h23 terminal will appear-------------

	a) Type the following in h1 terminal
		>> cd Project
		>> sudo python Traffic.py
		
		------------Normal traffic also will flow------------
		
OUTPUT>>
POX controller shows the statistics of the packet count and shows the list of blocked IP addresses