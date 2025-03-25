GeoEasy Python
==============

planes for Python rewrite of GeoEasy

Total redesign:

* change to OOP
* total separation of calculation functions from GUI, easy batch processing API
* coordinates are not part of the fieldbook, single coordinate source (may be more sources but separated from field books)
* data format to a standard format JSON or XML or ???, but geo,coo also supported
* SQLite database for points as one of the sources

Classes
-------

Base:
Point - id, east, north, elev, standard deviations, ...

Observations: 
direction - to, hz, v, sd, th, height-diff (triginometric), ...
levelling - from, to, height-diff, distance
distance - from, to, sd, v
Station - id, ih, oriention, ..., list of direction observations objects


Complex:
coordinate list
fieldbook different types for directions/levelling/distances 
totalstation observations list of station objects
traverse?
...

GUI
---

tkinter??? easy jump from tk or CustomTk from Tom Schimansky (pip install customtkinter)
Total redesign in Qt or wxPython??


