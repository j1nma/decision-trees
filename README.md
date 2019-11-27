# Decision Trees

This work has two components:

- Required is an implementation for sk-learn for a way to handle categorical data (and also mixed data, i.e. categorical and numerical). This implementation is hosted at this [fork](https://github.com/tcerdaITBA/scikit-learn). Error rate, information gain and gini index are provided in each result tree.

- Then, the solution is applied to the data sets provided.

- To visualise the results, the output of the tree is reported in a [graphical output](http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html).



The numeric parts of the user's matriculation number of the FH Technikum is used for deciding about the random seed for splitting your data into training & test set.

### Installation

At least for macOS environment:

```sh
export CC=/usr/bin/clang        
export CXX=/usr/bin/clang++
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib
```

```sh
$  pip3 install -r requirements.txt
```

### Running
```sh
$  python3 Exercise2.py
```