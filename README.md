# Adept 1.1
## Fast automatic differentiation library in C++

The Adept software library provides the capability to automatically
differentiate an algorithm written in C++.  It uses expression
templates in a way that allows it to compute adjoints and Jacobian
matrices significantly faster than the leading current tools that use
the same approach of operator overloading, and often not much slower
than hand-written adjoint code.

Note that this is not the latest version of Adept: if you want a
library that combines array features with automatic differentiation
then consider using [Adept 2.0](https://github.com/rjhogan/Adept-2).

For further information see:
* The [Adept web site](http://www.met.reading.ac.uk/clouds/adept/)
* A detailed [User Guide](http://www.met.reading.ac.uk/clouds/adept/adept_documentation_1.1.pdf)
* [A paper published in ACM TOMS](http://www.met.reading.ac.uk/~swrhgnrj/publications/adept.pdf) describing how it works.

To build Adept from a GitHub snapshot, do the following:

    autoreconf -fi

Then the normal make sequence:

    ./configure
    make
    make install