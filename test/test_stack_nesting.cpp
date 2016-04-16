/* test_stack_nesting - Acceleration of certain types of algorithm with nesting

  Copyright (C) 2016 European Centre for Medium Range Weather Forecasts

  Author: Robin Hogan <r.j.hogan@reading.ac.uk>

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

  This test case illustrates how Jacobian calculations can be
  accelerated in certain situations when they are composed of
  "modules" that each involve a few numbers being calculated from
  many. In this situation we may avoid redundant computations by using
  a nested stack to calculate a local Jacobian for each module, and
  then adding it to the main stack.

  Thanks to Julian Kaupe for suggesting this pattern.

*/

#include <iostream>
#include <vector>

#include "Timer.h"
#include "adept.h"

using adept::adouble;

// Arbitrary function with one output
static
adouble
many_in_one_out(const std::vector<adouble>& x, const adouble& yinit) {
  adouble y = yinit;
  for (int i = 0; i < x.size(); i += 2) {
    adouble t = x[i]*x[i];
    adouble e = exp(-x[i+1]);
    y = y*e + t*(1-e);
  }
  return y;
}

int
main(int argc, char** argv) {

  static const int N = 1024;

  adept::Stack s(4*N*N);

  // Independent variables
  std::vector<adouble> x(N), yinit(N);

  // Dependent variables
  std::vector<adouble> y(N);

  // Initialize timer
  Timer timer;
  timer.print_on_exit(true);
  int unnested_fwd = timer.new_activity("Unnested forward pass");
  int unnested_jac = timer.new_activity("Unnested Jacobian");
  int   nested_fwd = timer.new_activity(  "Nested forward pass");
  int   nested_jac = timer.new_activity(  "Nested Jacobian");

  std::vector<double> jac(N*N*2);

  std::cout << "*** UNNESTED ALGORITHM ***\n";

  s.new_recording();
  timer.start(unnested_fwd);
  for (int i = 0; i < N; ++i) {
    y[i] = many_in_one_out(x,yinit[i]);
  }

  timer.start(unnested_jac);
  s.clear_independents();
  s.clear_dependents();
  s.independent(x);
  s.independent(yinit);
  s.dependent(y);
  s.jacobian(&jac[0]);

  timer.stop();

  std::cout << s;

  std::cout << "*** NESTED ALGORITHM ***\n";

  double jac_nested[N+1];

  s.new_recording();
  timer.start(nested_fwd);
  for (int i = 0; i < N; ++i) {
    // Deactivate outer stack and start a nested one
    s.deactivate();

    {
      adept::Stack s_nested(4*N);

      // We need local variables registered with the nested stack
      std::vector<adouble> x_nested(N);
      adouble yinit_nested, y_nested;
      
      // Copy from the main variables - note that this puts references
      // to unregistered variables on the nested stack, but these are
      // deleted when we call new_recording
      for (int j = 0; j < N; ++j) {
	x_nested[j] = x[j];
      }
      yinit_nested = y[i];

      // Start with a fresh stack and only use local variables from
      // now on
      s_nested.new_recording();
      y_nested = many_in_one_out(x_nested, yinit_nested);

      s_nested.clear_independents();
      s_nested.clear_dependents();
      s_nested.independent(x_nested);
      s_nested.independent(yinit_nested);
      s_nested.dependent(y_nested);
      s_nested.jacobian(jac_nested);

      // Nested stack goes out of scope here
    }
    // Reactivate outer stack
    s.activate();

    // Copy nested Jacobian onto local stack - note that here we use
    // the original variables becase we know that x==x_nested and
    // yinit[i]==yinit_nested
    y[i].add_derivative_dependence(&x[0], jac_nested, N);
    y[i].append_derivative_dependence(yinit[i], jac_nested[N]);
  }

  timer.start(nested_jac);
  s.clear_independents();
  s.clear_dependents();
  s.independent(x);
  s.independent(yinit);
  s.dependent(y);
  s.jacobian(&jac[0]);

  timer.stop();

  std::cout << s;

  return 0;


}
