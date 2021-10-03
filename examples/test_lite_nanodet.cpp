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

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  nanodet->detect(img_bgr, detected_boxes, 0.3f);

  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete nanodet;

}



int main(__unused int argc, __unused char *argv[])
{
  test_nanodet();
  return 0;
}
