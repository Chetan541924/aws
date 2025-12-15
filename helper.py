def get_runtime_frame_helper():
    """
    Injected into generated scripts.
    Dynamically resolves the correct execution context (page or iframe)
    for EVERY action at runtime.
    """
    return r'''
def resolve_context(page, selector, timeout=5000):
    # Try main page first
    try:
        page.locator(selector).first.wait_for(timeout=1000)
        return page
    except:
        pass

    # Try all current frames (nested included)
    for frame in page.frames:
        try:
            frame.locator(selector).first.wait_for(timeout=1000)
            return frame
        except:
            continue

    raise Exception(f"Selector not found in page or any iframe: {selector}")
'''

generated_script = await gen_func(
    testcase_id=testcase_id,
    script_type=script_type,
    script_lang="python",
    testplan=testplan_dict,
    selected_madl_methods=None,
    logger=logger
)

# ----------------- INJECT IFRAME RUNTIME HELPER -----------------
runtime_helper = get_runtime_frame_helper()

if runtime_helper not in generated_script:
    generated_script = runtime_helper + "\n\n" + generated_script
