#ifndef TIME_LOGGER_H_
#define TIME_LOGGER_H_

#include <chrono>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
class TimeLogger{
private:
  std::chrono::system_clock::time_point start_;
  double time_;
  std::ostream* out_;
  bool print_;

public:
  TimeLogger(){
    time_ = 0;
    print_ = true;
  }

  TimeLogger(bool print){
    time_ = 0;
    print_ = print;
  }

  void start(){
//    start_ = std::chrono::system_clock::now();
    start_ = std::chrono::high_resolution_clock::now();
  }

  void start(const std::string& name){
    start_ = std::chrono::high_resolution_clock::now();

    if(print_){
      std::cout << name << std::flush;
    }
  }

  void stop(){
//    std::chrono::system_clock::time_point end
//      = std::chrono::system_clock::now();
    std::chrono::system_clock::time_point end
      = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt = end - start_;

    time_ = dt.count();

    if(print_){
//      std::cout << " time(sec) " << time_ << std::endl;
      print_time("");
    }
  }

  void print_time(const std::string& name){
    std::cout << name << " time(sec) " << time_ << std::endl;
  }

  double get_time() const{
    return time_;
  }

  static void print_current_time(std::ostream& o){
    auto current = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(current);

    o << std::ctime(&t);
  }

};

#endif // TIME_LOGGER_H_
