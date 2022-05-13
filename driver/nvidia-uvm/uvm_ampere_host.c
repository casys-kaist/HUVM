/*******************************************************************************
    Copyright (c) 2018 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_hal.h"
#include "uvm_global.h"
#include "uvm_user_channel.h"
#include "uvm_push_macros.h"
#include "hwref/ampere/ga100/dev_runlist.h"
#include "clc076.h"

void uvm_hal_ampere_host_clear_faulted_channel_register(uvm_user_channel_t *user_channel,
                                                        const uvm_fault_buffer_entry_t *fault)
{
    uvm_spin_loop_t spin;
    NvU32 channel_faulted_mask = 0;
    NvU32 clear_type_value = 0;

    UVM_ASSERT(!user_channel->gpu->parent->has_clear_faulted_channel_method);

    if (fault->fault_source.mmu_engine_type == UVM_MMU_ENGINE_TYPE_HOST) {
        clear_type_value = NV_CHRAM_CHANNEL_UPDATE_RESET_PBDMA_FAULTED;
        channel_faulted_mask = HWCONST(_CHRAM, CHANNEL, PBDMA_FAULTED, TRUE);
    }
    else if (fault->fault_source.mmu_engine_type == UVM_MMU_ENGINE_TYPE_CE) {
        clear_type_value = NV_CHRAM_CHANNEL_UPDATE_RESET_ENG_FAULTED;
        channel_faulted_mask = HWCONST(_CHRAM, CHANNEL, ENG_FAULTED, TRUE);
    }
    else {
        UVM_ASSERT_MSG(false, "Unsupported MMU engine type %s\n",
                       uvm_mmu_engine_type_string(fault->fault_source.mmu_engine_type));
    }

    // Wait for the channel to have the FAULTED bit set as this can race with
    // interrupt notification
    UVM_SPIN_WHILE(!(UVM_GPU_READ_ONCE(*user_channel->chram_channel_register) & channel_faulted_mask), &spin);

    UVM_GPU_WRITE_ONCE(*user_channel->chram_channel_register, clear_type_value);

    wmb();

    UVM_GPU_WRITE_ONCE(*user_channel->work_submission_offset, user_channel->work_submission_token);
}

static NvU32 instance_ptr_aperture_type_to_hw_value(uvm_aperture_t aperture)
{
    switch (aperture)
    {
        case UVM_APERTURE_SYS:
            return HWCONST(C076, CLEAR_FAULTED_A, INST_APERTURE, SYS_MEM_COHERENT);
        case UVM_APERTURE_VID:
            return HWCONST(C076, CLEAR_FAULTED_A, INST_APERTURE, VID_MEM);
        default:
            UVM_ASSERT_MSG(false, "Invalid aperture_type %d\n", aperture);
    }

    return 0;
}

static void instance_ptr_address_to_hw_values(NvU64 instance_ptr_address,
                                              NvU32 *instance_ptr_lo,
                                              NvU32 *instance_ptr_hi)
{
    // instance_ptr must be 4K aligned
    UVM_ASSERT_MSG(IS_ALIGNED(instance_ptr_address, 1 << 12), "instance_ptr 0x%llx\n", instance_ptr_address);
    instance_ptr_address >>= 12;

    *instance_ptr_lo = instance_ptr_address & HWMASK(C076, CLEAR_FAULTED_A, INST_LOW);
    *instance_ptr_hi = instance_ptr_address >> HWSIZE(C076, CLEAR_FAULTED_A, INST_LOW);
}

static NvU32 mmu_engine_type_to_hw_value(uvm_mmu_engine_type_t mmu_engine_type)
{
    switch (mmu_engine_type)
    {
        case UVM_MMU_ENGINE_TYPE_HOST:
            return HWCONST(C076, CLEAR_FAULTED_A, TYPE, PBDMA_FAULTED);
        case UVM_MMU_ENGINE_TYPE_CE:
            return HWCONST(C076, CLEAR_FAULTED_A, TYPE, ENG_FAULTED);
        default:
            UVM_ASSERT_MSG(false, "Unsupported MMU engine type %s\n",
                       uvm_mmu_engine_type_string(mmu_engine_type));
    }

    return 0;
}

void uvm_hal_ampere_host_clear_faulted_channel_sw_method(uvm_push_t *push,
                                                         uvm_user_channel_t *user_channel,
                                                         const uvm_fault_buffer_entry_t *fault)
{
    NvU32 clear_type_value;
    NvU32 aperture_type_value;
    NvU32 instance_ptr_lo, instance_ptr_hi;
    uvm_gpu_phys_address_t instance_ptr = user_channel->instance_ptr.addr;

    UVM_ASSERT(user_channel->gpu->parent->has_clear_faulted_channel_sw_method);

    clear_type_value = mmu_engine_type_to_hw_value(fault->fault_source.mmu_engine_type);
    aperture_type_value = instance_ptr_aperture_type_to_hw_value(instance_ptr.aperture);

    instance_ptr_address_to_hw_values(instance_ptr.address, &instance_ptr_lo, &instance_ptr_hi);

    NV_PUSH_2U(C076, CLEAR_FAULTED_A, HWVALUE(C076, CLEAR_FAULTED_A, INST_LOW, instance_ptr_lo) |
                                      aperture_type_value |
                                      clear_type_value,
                     CLEAR_FAULTED_B, HWVALUE(C076, CLEAR_FAULTED_B, INST_HI, instance_ptr_hi));
}
