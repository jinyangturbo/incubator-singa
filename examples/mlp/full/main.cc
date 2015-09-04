#include <string>
#include "singa.h"
#include "hidden_layer.h"
#include "myproto.pb.h"
#include "utils/common.h"

int main(int argc, char **argv) {
  //  must create driver at the beginning and call its Init method.
  singa::Driver driver;
  driver.Init(argc, argv);

  //  if -resume in argument list, set resume to true; otherwise false
  int resume_pos = singa::ArgPos(argc, argv, "-resume");
  bool resume = (resume_pos != -1);

  //  users can register new subclasses of layer, updater, etc.
  driver.RegisterLayer<singa::HiddenLayer, std::string>("kHidden");

  //  get the job conf, and custmize it if need
  singa::JobProto jobConf = driver.job_conf();

  //  submit the job
  driver.Submit(resume, jobConf);
  return 0;
}
