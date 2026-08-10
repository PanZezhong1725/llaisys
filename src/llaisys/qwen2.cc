#include "llaisys/qwen2.h"
#include "llaisys_tensor.hpp"
#include "qwen2_model.hpp"

#include <string>

__C {

    llaisysQwen2Model_t qwen2Create(
        llaisysDeviceType_t device_type,
        int device_id) {
        return reinterpret_cast<llaisysQwen2Model_t>(
            new llaisys::models::Qwen2Model(device_type, device_id));
    }

    void qwen2Destroy(
        llaisysQwen2Model_t model) {
        delete reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    }

    void qwen2LoadWeight(
        llaisysQwen2Model_t model,
        const char *name,
        llaisysTensor_t weight) {
        auto *m = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        m->loadWeight(std::string(name), weight->tensor);
    }

    void qwen2Forward(
        llaisysQwen2Model_t model,
        llaisysTensor_t input_ids,
        llaisysTensor_t output_logits) {
        auto *m = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        m->forward(input_ids->tensor, output_logits->tensor);
    }

    void qwen2ResetKV(
        llaisysQwen2Model_t model) {
        auto *m = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        m->resetKV();
    }
}
