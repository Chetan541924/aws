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




async def wait_for_ccs_render(frame, timeout=30000):
    await frame.wait_for_function(
        """
        () => {
            const body = document.body;
            if (!body) return false;
            return body.innerText && body.innerText.trim().length > 0;
        }
        """,
        timeout=timeout
    )



async def assert_text_visible(frame, expected_text: str):
    # 🔴 WAIT FOR CCS TO ACTUALLY RENDER
    await wait_for_ccs_render(frame)

    # Try multiple strategies (legacy-safe)
    strategies = [
        lambda: frame.get_by_text(expected_text, exact=False),
        lambda: frame.locator(f"text={expected_text}"),
        lambda: frame.locator(f"//*[contains(text(), '{expected_text}')]"),
    ]

    for strategy in strategies:
        try:
            locator = strategy()
            await locator.wait_for(state="visible", timeout=5000)
            return
        except Exception:
            continue

    # If nothing worked, dump useful body HTML
    body_html = await frame.evaluate(
        "() => document.body ? document.body.innerText : '<NO BODY>'"
    )

    raise RuntimeError(
        f"Text not found using any strategy: '{expected_text}'\n"
        f"Rendered body text:\n{body_html[:2000]}"
    )







