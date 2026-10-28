(() => {
  "use strict";
  const API_ROOT = "/api/v1";
  const MAX_FILE_SIZE = 10 * 1024 * 1024, MAX_SEED = Number.MAX_SAFE_INTEGER;
  const POLL_INTERVAL = 500, POLL_TIMEOUT = 120000;
  const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const SUCCESS = new Set(["success", "succeeded", "completed"]);
  const FAILURE = new Set(["failed", "error"]);
  const CANCELLED = new Set(["cancelled", "canceled"]);
  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("generator"), apiKey: $("api-key"), file: $("file-input"),
    dropZone: $("drop-zone"), fileName: $("file-name"), fileError: $("file-error"),
    preview: $("preview-container"), previewImage: $("preview-image"), removeFile: $("remove-file"),
    style: $("style-select"), styleDescription: $("style-description"), turbo: $("mode-turbo"),
    seed: $("seed-input"), generate: $("generate-button"), cancel: $("cancel-button"),
    refresh: $("refresh-button"), formError: $("form-error"), stage: $("stage-text"),
    progressValue: $("progress-value"), progress: $("progress"), taskStatus: $("task-status"),
    taskId: $("task-id"), result: $("result-section"), resultImage: $("result-image"),
    download: $("download-link")
  };
  let styles = [], selectedFile = null, previewUrl = null, resultUrl = null;
  let activeTaskId = null, taskActive = false, cancelRequested = false;
  let submitting = false, cancelling = false, manualLoading = false, resultLoading = false;
  let pollSession = 0, taskRevision = 0;
  let pollTimer = null, timeoutTimer = null, pollController = null;

  function setFormError(message) { el.formError.textContent = message || ""; }
  function requireApiKey() {
    const key = el.apiKey.value.trim();
    if (!key) throw new Error("请输入 API Key 后再继续。");
    return key;
  }
  function taskPath(id, suffix = "") { return `${API_ROOT}/tasks/${encodeURIComponent(id)}${suffix}`; }
  function httpError(status) {
    if (status === 400) return new Error("请求未被接受，请检查图片、风格、模式和种子。");
    if (status === 401 || status === 403) return new Error("API Key 无效或权限不足，请检查后重试。");
    if (status === 404) return new Error("请求的资源不存在或已过期，请刷新状态或重新生成。");
    if (status === 409) return new Error("任务状态已变化，请查询当前任务后再重试。");
    if (status === 413) return new Error("图片超过服务限制，请选择不超过 10 MiB 的文件。");
    if (status === 429) return new Error("请求过于频繁，请稍等片刻后重试。");
    if (status >= 500) return new Error("服务暂时不可用，请稍后重试；若持续失败请联系管理员。");
    return new Error(`服务返回错误（HTTP ${status}），请稍后重试。`);
  }
  async function guardedFetch(url, options) {
    try { return await fetch(url, options); }
    catch (error) {
      if (error.name === "AbortError") throw error;
      throw new Error("无法连接服务，请检查网络连接，并确认后端服务可用。");
    }
  }
  async function apiFetch(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("X-API-Key", el.apiKey.value.trim());
    const response = await guardedFetch(path, { ...options, headers });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      if (body.error && typeof body.error.message === "string") throw new Error(body.error.message);
      throw httpError(response.status);
    }
    return response;
  }
  async function fetchJson(url, options) {
    const response = await apiFetch(url, options);
    try { return await response.json(); }
    catch { throw new Error("服务响应格式异常，请稍后重试或联系管理员。"); }
  }
  function updateControls() {
    const busy = submitting || taskActive || resultLoading || manualLoading || cancelling;
    el.generate.disabled = busy || !selectedFile || el.style.disabled || !el.style.value;
    el.generate.textContent = submitting ? "正在提交…" : "开始生成";
    el.cancel.disabled = !activeTaskId || !taskActive || cancelRequested || cancelling || manualLoading;
    el.cancel.textContent = cancelling ? "正在请求取消…" : "取消当前任务";
    el.refresh.disabled = !activeTaskId || manualLoading || resultLoading || cancelling;
  }
  function updateStyleDescription() {
    const item = styles.find((entry) => entry.key === el.style.value);
    el.styleDescription.textContent = item ? item.description : "请选择一种艺术风格。";
  }
  async function loadStyles() {
    try {
      const data = await fetchJson(`${API_ROOT}/styles`);
      if (!data || !Array.isArray(data.styles)) throw new Error("风格数据格式异常，请联系管理员。");
      styles = data.styles.map((item) => ({
        key: String(item && item.key != null ? item.key : ""),
        label: String(item && item.label != null ? item.label : item && item.key != null ? item.key : ""),
        description: String(item && item.description ? item.description : "暂无风格说明。")
      })).filter((item) => item.key);
      if (!styles.length) throw new Error("服务暂未提供可用风格，请稍后刷新重试。");
      const fragment = document.createDocumentFragment();
      for (const item of styles) {
        const option = document.createElement("option");
        option.value = item.key; option.textContent = item.label;
        fragment.appendChild(option);
      }
      el.style.replaceChildren(fragment);
      el.style.disabled = false;
      updateStyleDescription();
    } catch (error) {
      styles = [];
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "风格加载失败";
      el.style.replaceChildren(option);
      el.style.disabled = true;
      el.styleDescription.textContent = "请检查网络与服务状态后刷新页面重试。";
      setFormError(`风格加载失败：${error.message}`);
    } finally { updateControls(); }
  }
  function revokePreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; el.previewImage.removeAttribute("src");
  }
  function clearResult() {
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = null;
    el.resultImage.removeAttribute("src");
    el.download.removeAttribute("href");
    el.result.hidden = true;
  }
  function clearSelectedFile() {
    selectedFile = null;
    el.file.value = "";
    el.fileName.textContent = "尚未选择照片"; el.fileError.textContent = "";
    el.preview.hidden = true; revokePreview();
    updateControls();
  }
  function rejectFile(message) {
    clearSelectedFile();
    el.fileError.textContent = message;
  }
  function selectFile(file) {
    clearSelectedFile();
    if (!file) return;
    if (!IMAGE_TYPES.has(file.type.toLowerCase())) return rejectFile("仅支持 JPEG、PNG、WebP 图片，请重新选择。");
    if (file.size === 0) return rejectFile("所选文件为空，请选择有效的图片文件。");
    if (file.size > MAX_FILE_SIZE) return rejectFile("图片不能超过 10 MiB，请压缩或更换图片后重试。");
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    el.previewImage.src = previewUrl; el.preview.hidden = false;
    el.fileName.textContent = `${file.name}（${(file.size / 1024 / 1024).toFixed(2)} MiB）`;
    updateControls();
  }
  function readSeed() {
    const raw = el.seed.value.trim();
    if (!raw && !el.seed.validity.badInput) return null;
    if (!/^\d+$/.test(raw)) throw new Error("随机种子必须是 0 至 2^53−1 之间的整数。");
    const value = Number(raw);
    if (!Number.isSafeInteger(value) || value < 0 || value > MAX_SEED)
      throw new Error("随机种子超出范围，请输入 0 至 9007199254740991 之间的整数。");
    return value;
  }
  function statusLabel(status) {
    return {
      pending: "等待服务处理", queued: "排队等待处理", running: "正在生成",
      processing: "正在生成", success: "生成完成", succeeded: "生成完成",
      completed: "生成完成", failed: "生成失败", error: "生成失败",
      cancelled: "任务已取消", canceled: "任务已取消"
    }[status] || `任务状态：${status || "未知"}`;
  }
  function taskIdOf(task) { return task && task.task_id != null ? String(task.task_id) : ""; }
  function statusOf(task) { return String(task.status || "").toLowerCase(); }
  function taskErrorMessage(error) {
    const structured = error && typeof error === "object" && !Array.isArray(error);
    const message = typeof error === "string" ? error : structured ? error.message : "";
    return typeof message === "string" && message.trim() ? message.trim() : "请调整照片或参数后重试。";
  }
  function isCurrent(session, id = activeTaskId) { return session === pollSession && Boolean(id) && id === activeTaskId; }
  function currentResponse(task, session, revision) { return isCurrent(session) && revision === taskRevision && taskIdOf(task) === activeTaskId; }
  function renderTask(task) {
    const status = statusOf(task);
    const progress = task.progress && typeof task.progress === "object" ? task.progress : {};
    const value = Number(progress.value), max = Number(progress.max);
    cancelRequested = Boolean(task.cancel_requested);
    el.taskId.textContent = `任务 ID：${activeTaskId}`;
    el.stage.textContent = String(progress.phase || statusLabel(status));
    if (Number.isFinite(value) && Number.isFinite(max) && max > 0) {
      el.progress.max = max;
      el.progress.value = Math.max(0, Math.min(value, max));
      el.progressValue.textContent = `${value} / ${max}`;
    } else {
      el.progress.max = 100;
      el.progress.value = 0;
      el.progressValue.textContent = "";
    }
    if (SUCCESS.has(status)) el.taskStatus.textContent = "任务已完成，正在准备 PNG 图片。";
    else if (FAILURE.has(status)) el.taskStatus.textContent = `生成失败：${taskErrorMessage(task.error)}`;
    else if (CANCELLED.has(status)) el.taskStatus.textContent = "任务已由服务确认取消。";
    else if (cancelRequested) el.taskStatus.textContent = "已请求取消，等待服务确认；后台任务可能仍在运行。";
    else el.taskStatus.textContent = statusLabel(status);
  }
  function stopPolling() {
    if (pollTimer !== null) clearTimeout(pollTimer);
    if (timeoutTimer !== null) clearTimeout(timeoutTimer);
    pollTimer = timeoutTimer = null;
    const controller = pollController; pollController = null;
    if (controller) controller.abort();
  }
  function applyTask(task, session) {
    if (!isCurrent(session) || taskIdOf(task) !== activeTaskId) return false;
    const status = statusOf(task);
    const terminal = SUCCESS.has(status) || FAILURE.has(status) || CANCELLED.has(status);
    if (!taskActive && !terminal) return true;
    renderTask(task);
    if (SUCCESS.has(status)) {
      taskActive = false;
      stopPolling();
      el.refresh.hidden = true;
      void loadResult(session);
    } else if (terminal) {
      taskActive = false;
      stopPolling();
      el.refresh.hidden = true;
    }
    updateControls();
    return terminal;
  }
  async function pollTask(session) {
    pollTimer = null;
    if (!isCurrent(session) || !taskActive || pollController) return;
    const revision = taskRevision, controller = new AbortController();
    pollController = controller;
    try {
      const task = await fetchJson(taskPath(activeTaskId), { signal: controller.signal });
      if (!isCurrent(session)) return;
      if (currentResponse(task, session, revision)) {
        if (el.formError.textContent.startsWith("状态查询失败：")) setFormError("");
        if (applyTask(task, session)) return;
      }
    } catch (error) {
      if (error.name === "AbortError" || !isCurrent(session)) return;
      if (revision === taskRevision) {
        setFormError(`状态查询失败：${error.message} 页面将自动重试；后台任务可能仍在运行。`);
        el.taskStatus.textContent = "暂时无法获取进度，将自动重试；这不表示任务已停止或取消。";
      }
    } finally {
      if (pollController === controller) pollController = null;
    }
    if (isCurrent(session) && taskActive) {
      pollTimer = setTimeout(() => void pollTask(session), POLL_INTERVAL);
    }
  }
  function pollingTimedOut(session) {
    if (!isCurrent(session) || !taskActive) return;
    stopPolling();
    el.stage.textContent = "自动查询已暂停";
    el.taskStatus.textContent = "自动查询已达 120 秒；这不表示后台任务已停止或取消。请点击“查询当前任务”手动获取状态。";
    el.refresh.hidden = false;
    updateControls();
  }
  function startPolling(session) {
    el.refresh.hidden = true;
    timeoutTimer = setTimeout(() => pollingTimedOut(session), POLL_TIMEOUT);
    void pollTask(session);
  }
  async function loadResult(session) {
    if (!isCurrent(session)) return;
    resultLoading = true;
    el.stage.textContent = "正在下载结果";
    updateControls();
    try {
      const response = await apiFetch(taskPath(activeTaskId, "/result"));
      let blob;
      try { blob = await response.blob(); }
      catch { throw new Error("图片下载中断，请检查网络后点击“查询当前任务”重试。"); }
      if (!blob.size) throw new Error("服务返回了空图片，请点击“查询当前任务”重试或联系管理员。");
      if (!isCurrent(session)) return;
      clearResult();
      resultUrl = URL.createObjectURL(blob);
      el.resultImage.src = resultUrl; el.download.href = resultUrl;
      el.result.hidden = false;
      el.stage.textContent = "生成完成"; el.taskStatus.textContent = "生成完成，可预览或下载 PNG 图片。";
      setFormError("");
    } catch (error) {
      if (!isCurrent(session) || error.name === "AbortError") return;
      el.taskStatus.textContent = "任务已完成，但图片尚未下载成功；后台结果不会因页面超时而自动取消。";
      el.refresh.hidden = false;
      setFormError(`结果获取失败：${error.message} 可点击“查询当前任务”重试。`);
    } finally {
      if (isCurrent(session)) { resultLoading = false; updateControls(); }
    }
  }
  async function submitTask() {
    if (submitting || taskActive || resultLoading || manualLoading || cancelling) return;
    setFormError("");
    let seed;
    try {
      requireApiKey();
      if (!selectedFile) throw new Error("请先选择一张符合要求的人像照片。");
      if (!el.style.value || el.style.disabled) throw new Error("请选择艺术风格；若列表为空，请刷新页面重试。");
      seed = readSeed();
    } catch (error) {
      setFormError(error.message);
      if (el.formError.textContent.includes("种子")) el.seed.focus();
      return;
    }
    stopPolling();
    const session = ++pollSession;
    taskRevision += 1;
    activeTaskId = null; taskActive = cancelRequested = false;
    clearResult();
    el.taskId.textContent = ""; el.refresh.hidden = true; submitting = true;
    el.stage.textContent = "正在提交任务";
    el.progress.max = 100; el.progress.value = 0; el.progressValue.textContent = "";
    el.taskStatus.textContent = "正在上传照片并创建任务，请稍候。";
    updateControls();
    const data = new FormData();
    data.append("file", selectedFile);
    data.append("style", el.style.value);
    data.append("mode", el.turbo.checked ? "turbo" : "standard");
    if (seed !== null) data.append("seed", String(seed));
    try {
      const task = await fetchJson(`${API_ROOT}/tasks`, { method: "POST", body: data });
      const id = taskIdOf(task);
      if (!id) throw new Error("服务未返回任务 ID，请稍后重试或联系管理员。");
      activeTaskId = id;
      taskActive = true;
      el.taskId.textContent = `任务 ID：${id}`;
      if (!applyTask(task, session) && isCurrent(session)) startPolling(session);
    } catch (error) {
      if (session === pollSession) {
        setFormError(`任务提交失败：${error.message} 若网络中断，任务是否已创建可能未知，请核对服务状态后再重试。`);
        el.stage.textContent = "提交未完成";
        el.taskStatus.textContent = "请根据上方提示检查网络、密钥或输入后重试。";
      }
    } finally {
      submitting = false;
      updateControls();
    }
  }
  async function refreshTask() {
    if (!activeTaskId || manualLoading || resultLoading || cancelling || pollController || timeoutTimer !== null) return;
    setFormError("");
    manualLoading = true;
    updateControls();
    const session = pollSession, revision = taskRevision;
    try {
      const task = await fetchJson(taskPath(activeTaskId));
      if (!currentResponse(task, session, revision)) return;
      if (!applyTask(task, session) && taskActive) {
        el.taskStatus.textContent = `${statusLabel(statusOf(task))}；自动查询仍暂停，后台任务可能仍在运行，可稍后再次手动查询。`;
      }
    } catch (error) {
      if (!isCurrent(session)) return;
      setFormError(`手动查询失败：${error.message} 后台任务状态未知，请检查网络后重试；这不代表任务已取消。`);
      el.refresh.hidden = false;
    } finally {
      if (isCurrent(session)) { manualLoading = false; updateControls(); }
    }
  }
  async function cancelTask() {
    if (!activeTaskId || !taskActive || cancelling || cancelRequested || manualLoading) return;
    const session = pollSession, id = activeTaskId;
    taskRevision += 1;
    const revision = taskRevision;
    cancelling = true;
    setFormError("");
    el.taskStatus.textContent = "正在请求取消，需等待服务确认；后台任务可能仍在运行。";
    updateControls();
    try {
      const task = await fetchJson(taskPath(id, "/cancel"), { method: "POST" });
      if (!currentResponse(task, session, revision)) return;
      applyTask(task, session);
      if (taskActive) el.taskStatus.textContent = "取消请求已发送，等待服务确认；请勿将其视为已停止。";
    } catch (error) {
      if (!isCurrent(session, id) || !taskActive) return;
      setFormError(`取消请求失败：${error.message} 请检查网络后重试；后台任务可能仍在运行。`);
      el.taskStatus.textContent = "取消尚未得到服务确认，请重试或查询当前任务状态。";
    } finally {
      if (isCurrent(session, id)) { cancelling = false; updateControls(); }
    }
  }
  function bindEvents() {
    el.apiKey.addEventListener("input", () => {
      try {
        if (el.apiKey.value) sessionStorage.setItem("aigc.apiKey", el.apiKey.value);
        else sessionStorage.removeItem("aigc.apiKey");
      } catch { setFormError("无法写入会话存储；请检查浏览器隐私设置，API Key 仅在当前页面有效。"); }
    });
    el.file.addEventListener("change", () => selectFile(el.file.files[0]));
    el.removeFile.addEventListener("click", clearSelectedFile);
    for (const type of ["dragenter", "dragover"]) {
      el.dropZone.addEventListener(type, (event) => {
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        el.dropZone.classList.add("drag-over");
      });
    }
    for (const type of ["dragleave", "drop"])
      el.dropZone.addEventListener(type, () => el.dropZone.classList.remove("drag-over"));
    el.dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) return;
      if (files.length > 1) rejectFile("一次只能上传一张照片，请重新拖放单个文件。");
      else selectFile(files[0]);
    });
    el.style.addEventListener("change", () => { updateStyleDescription(); updateControls(); });
    el.form.addEventListener("submit", (event) => { event.preventDefault(); void submitTask(); });
    el.cancel.addEventListener("click", () => void cancelTask());
    el.refresh.addEventListener("click", () => void refreshTask());
  }
  try { el.apiKey.value = sessionStorage.getItem("aigc.apiKey") || ""; }
  catch { setFormError("无法读取会话存储；请检查浏览器隐私设置后重新输入 API Key。"); }
  bindEvents();
  updateControls();
  void loadStyles();
})();
