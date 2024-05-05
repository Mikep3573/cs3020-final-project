Michael Piscione and Dylan Laberge <br>
We planned on implementing Class Objects and Methods into the compiler. We edited both the python_pretty_printer and the python_parser files in order to support class methods. 
We created two new passes called “unnestify” and “compile dataclasses”. Unnestify runs before all the other passes and moves all the methods of an object outside of the object to the top-level. 
Compile dataclasses removes all class definitions, replaces class constructor calls with tuple declarations, and replaces class field arguments with tuple subscripts. There is a limitation as 
well that all method “self” parameters must be uniquely named (e.g., self1, self2, self3, etc). We also cannot edit object fields after they have been created. <br>
We were able to get the program to output through all the passes on most of the test cases. Without adding methods we got some of the programs to output the correct result 
through run_tests however we could not get run_tests to fully run any of the programs with methods. 
