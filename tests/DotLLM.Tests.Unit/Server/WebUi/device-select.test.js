// Run by DeviceSelectJsTests (node). Exits non-zero on the first failed assertion. Tests the pure logic in
// src/DotLLM.Server/wwwroot/device-select.js (#1704).
const assert = require('node:assert/strict');
const path = require('node:path');
const DS = require(path.resolve(process.argv[2]));

const GIB = 1024 ** 3;
const payload = {
    recommended_device: 'gpu:0',
    backends: [
        { name: 'cpu', servable: true, devices: [{ index: 0, name: 'x64 CPU', device_string: 'cpu' }] },
        { name: 'cuda', servable: true, devices: [{ index: 0, name: 'RTX 3060', device_string: 'gpu:0', total_memory_bytes: 12 * GIB, free_memory_bytes: 1 * GIB }] },
        {
            name: 'vulkan', servable: true, devices: [
                { index: 0, name: 'NVIDIA RTX 3060', device_string: 'vulkan:0', total_memory_bytes: 12 * GIB },
                { index: 1, name: 'AMD Radeon 8060S', device_string: 'vulkan:1', total_memory_bytes: 64 * GIB, is_default: true },
            ],
        },
    ],
};

// every device is listed, CPU first, with name and memory
const choices = DS.buildChoices(payload);
assert.deepEqual(choices.map(c => c.value), ['cpu', 'gpu:0', 'vulkan:0', 'vulkan:1']);
assert.match(choices[1].label, /RTX 3060/);
assert.match(choices[1].label, /1\.0 GiB free \/ 12 GiB/);   // free/total when known
assert.match(choices[3].label, /Radeon 8060S/);
assert.match(choices[3].label, /64 GiB/);                    // total only when free is unknown
assert.doesNotMatch(choices[3].label, /free/);

// unservable backends are not offered
const unservable = DS.buildChoices({ backends: [{ name: 'vulkan', servable: false, devices: [{ index: 0, name: 'v', device_string: 'vulkan:0' }] }] });
assert.deepEqual(unservable.map(c => c.value), ['cpu']);

// default = recommended; falls back to first GPU; cpu when none
assert.equal(DS.defaultValue(choices, payload), 'gpu:0');
assert.equal(DS.defaultValue(choices, { ...payload, recommended_device: 'vulkan:1' }), 'vulkan:1');
assert.equal(DS.defaultValue(choices, { ...payload, recommended_device: 'vulkan' }), 'vulkan:1');   // bare "vulkan" -> the is_default entry
assert.equal(DS.defaultValue(choices, { ...payload, recommended_device: undefined }), 'gpu:0');
assert.equal(DS.defaultValue(DS.buildChoices({ backends: [] }), {}), 'cpu');

// a user's choice survives a refresh whose recommendation differs (CUDA full -> recommended moves to vulkan)
const refreshed = { ...payload, recommended_device: 'vulkan:1' };
assert.equal(DS.resolveSelection(DS.buildChoices(refreshed), 'gpu:0', true, refreshed), 'gpu:0');
assert.equal(DS.resolveSelection(DS.buildChoices(refreshed), 'vulkan:0', true, refreshed), 'vulkan:0');
// untouched: the recommendation applies
assert.equal(DS.resolveSelection(DS.buildChoices(refreshed), 'gpu:0', false, refreshed), 'vulkan:1');
// touched, but the chosen device vanished: fall back to the default rather than keep a dead value
assert.equal(DS.resolveSelection(DS.buildChoices(refreshed), 'gpu:7', true, refreshed), 'vulkan:1');

// layer offload is CUDA-only
assert.equal(DS.supportsLayerOffload('gpu:0'), true);
assert.equal(DS.supportsLayerOffload('gpu'), true);
assert.equal(DS.supportsLayerOffload('vulkan:1'), false);
assert.equal(DS.supportsLayerOffload('vulkan'), false);
assert.equal(DS.supportsLayerOffload('cpu'), false);

console.log('ok');
