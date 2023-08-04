#include <fl/Headers.h>
#include <iostream>

using namespace fl;
using namespace std;

int main(int argc, char const *argv[])
{
  Engine* engine = FllImporter().fromFile("fuzzy.fll");

  std::string status;
  if (not engine->isReady(&status))
      throw Exception("[engine error] engine is not ready:n" + status, FL_AT);

  InputVariable* snr = engine->getInputVariable("snr");
  OutputVariable* tp = engine->getOutputVariable("tp");
  OutputVariable* sf = engine->getOutputVariable("tp");

  snr->setValue(20);
  engine->process();

  cout << "TP=" << Op::str(tp->getValue()) <<
          "SF=" << Op::str(sf->getValue());

  delete engine;
  return 0;
}
