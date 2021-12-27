//
// Created by DefTruth on 2021/10/2.
//

#include "lite/lite.h"

static void test_nanodet()
{
  std::string onnx_path = "../hub/onnx/cv/nanodet_m.onnx";
  std::string test_img_path = "../resources/9.jpg";
  std::string save_img_path = "../logs/9.jpg";

  auto *nanodet = new lite::cv::detection::NanoDet(onnx_path); 

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  nanodet->detect(img_bgr, detected_boxes, 0.3f);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NanoDet Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete nanodet;

}

static void test_nanodet_plus()
{
  std::string onnx_path = "../hub/onnx/cv/nanodet-plus-m-1.5x_416.onnx";
  std::string test_img_path = "../resources/9.jpg";
  std::string save_img_path = "../logs/9_plus.jpg";

  auto *nanodet_plus = new lite::cv::detection::NanoDetPlus(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  nanodet_plus->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NanoDetPlus Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete nanodet_plus;

}


int main(__unused int argc, __unused char *argv[])
{
  test_nanodet();
  test_nanodet_plus();
  return 0;
}
