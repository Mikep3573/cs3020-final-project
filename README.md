*Authors:*<br>
&emsp;Michael Piscione <br>
&emsp;Dylan Laberge

# Approach to Implementation

----

&emsp;We planned on implementing Class Objects and Methods into the compiler. We edited both the python_pretty_printer and the 
python_parser files in order to support class methods. We created two new passes called “unnestify” and “compile dataclasses”. 
Unnestify runs before all the other passes and moves all the methods of a class outside of the class definition (to the top-level). 
Compile dataclasses removes all class definitions, replaces class constructor calls with tuple declarations, and replaces 
class field arguments with tuple subscripts. Other than that, most of our time was spent updating the existing passes for class methods and fixing
errors as they appeared. There are a few limitations as well. Namely, all method “self” parameters 
must be uniquely named (e.g., self1, self2, self3, etc), and the programmer cannot edit the fields of an individual class instance once created.



# Unimplemented Features

---

&emsp;We were able to get the program to output through all the passes on most of the test cases. Without adding methods we 
got most of the programs to output the correct result through run_tests however we could not get run_tests to fully 
run any of the programs with methods. We believe at least part of the problem is due to an incorrect implementation
of function pointers in tuples (class representation). Furthermore, we likely were over ambitious with some of the later
test cases, namely: test10, and test11. Beyond these, there are a number of errors we did not have the time to fully
investigate.

Features we could've added if we had more time:
1. Class Inheritence
2. For Loops (something we were initially talking about)
3. Iterator objects (in 'for' loops)
