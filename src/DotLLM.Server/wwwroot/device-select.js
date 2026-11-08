// Pure device-choice logic for the model-load modal (#1704). No DOM access, so it can be unit-tested under node
// (tests/DotLLM.Tests.Unit/Server/WebUi/device-select.test.js). Loaded before app.js; exposes globalThis.DeviceSelect.
(function (root) {
    const GIB = 1024 * 1024 * 1024;

    function gib(bytes) {
        const v = bytes / GIB;
        return (v >= 10 ? v.toFixed(0) : v.toFixed(1)) + ' GiB';
    }

    /** CUDA "gpu:N" devices support partial layer offload; Vulkan (and CPU) do not. */
    function supportsLayerOffload(value) {
        return /^(gpu|cuda)(:\d+)?$/i.test(value || '');
    }

    function memoryText(d) {
        if (d.free_memory_bytes != null && d.total_memory_bytes != null)
            return `${gib(d.free_memory_bytes)} free / ${gib(d.total_memory_bytes)}`;
        if (d.total_memory_bytes != null) return `${gib(d.total_memory_bytes)}`;
        return '';
    }

    /**
     * Every selectable device from a /v1/devices payload: CPU first, then each servable GPU device in backend order.
     * Each choice: { value (the device string to send), label, kind, supportsLayers, isDefault }.
     */
    function buildChoices(devices) {
        const choices = [{ value: 'cpu', label: 'CPU', kind: 'cpu', supportsLayers: false, isDefault: false }];
        for (const b of (devices && devices.backends) || []) {
            if (b.name === 'cpu' || !b.servable) continue;
            for (const d of b.devices || []) {
                if (!d.device_string) continue;
                const mem = memoryText(d);
                choices.push({
                    value: d.device_string,
                    label: `${d.name} [${d.device_string}${b.name === 'cuda' ? ', CUDA' : ', Vulkan'}]` + (mem ? ` - ${mem}` : ''),
                    kind: b.name,
                    supportsLayers: supportsLayerOffload(d.device_string),
                    isDefault: d.is_default === true,
                });
            }
        }
        return choices;
    }

    /** The preselected value: the server's recommended device when listed (a bare "vulkan" maps to the default Vulkan entry), else the first GPU, else cpu. */
    function defaultValue(choices, devices) {
        const rec = devices && devices.recommended_device;
        if (rec && rec !== 'cpu') {
            if (choices.some(c => c.value === rec)) return rec;
            if (rec === 'vulkan') {
                const v = choices.find(c => c.kind === 'vulkan' && c.isDefault) || choices.find(c => c.kind === 'vulkan');
                if (v) return v.value;
            }
        }
        const gpu = choices.find(c => c.kind !== 'cpu');
        return gpu ? gpu.value : 'cpu';
    }

    /** The value to show: a user's own choice is kept while it is still listed (never overridden by a refresh); otherwise the default. */
    function resolveSelection(choices, current, touched, devices) {
        if (touched && current && choices.some(c => c.value === current)) return current;
        return defaultValue(choices, devices);
    }

    const api = { buildChoices, defaultValue, resolveSelection, supportsLayerOffload };
    root.DeviceSelect = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
