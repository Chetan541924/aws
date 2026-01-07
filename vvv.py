elif action_type == "RADIO":
    step_lower = step_name.lower()

    # Extract answer index (answer = 1 / 2)
    match = re.search(r"answer\s*=\s*(\d+)", step_lower)
    if not match:
        raise RuntimeError("RADIO step must specify answer = <number>")

    answer_index = match.group(1)

    # Resolve frames
    nav_frame, content_frame = resolve_ccs_frames(page)

    # ============================================================
    # EXTENDED STABILIZATION
    # ============================================================
    logger.info(LogCategory.EXECUTION, "[DEBUG] Waiting for page to stabilize...")
    await page.wait_for_timeout(3000)
    await content_frame.wait_for_load_state("networkidle", timeout=15000)
    await page.wait_for_timeout(2000)

    # ============================================================
    # STRATEGY: Find ALL radio groups on the page, select by index
    # ============================================================
    
    # Find all visible radio button groups (by unique 'name' attribute)
    radio_groups = await content_frame.evaluate("""
        () => {
            const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
            const groups = {};
            
            radios.forEach(radio => {
                const name = radio.name;
                if (!groups[name]) {
                    groups[name] = [];
                }
                groups[name].push({
                    id: radio.id,
                    name: radio.name,
                    value: radio.value,
                    checked: radio.checked,
                    visible: radio.offsetParent !== null
                });
            });
            
            return groups;
        }
    """)

    logger.info(
        LogCategory.EXECUTION,
        f"[DEBUG] Found {len(radio_groups)} radio button groups: {list(radio_groups.keys())}"
    )

    # ============================================================
    # Find the MOST RECENT visible radio group (heuristic)
    # ============================================================
    target_group_name = None
    target_group_radios = []

    # Get the last (most recent) visible radio group
    for group_name, radios in radio_groups.items():
        visible_radios = [r for r in radios if r['visible']]
        if visible_radios:
            target_group_name = group_name
            target_group_radios = visible_radios

    if not target_group_name:
        raise RuntimeError("No visible radio button groups found on page")

    logger.info(
        LogCategory.EXECUTION,
        f"[DEBUG] Selected radio group: {target_group_name} with {len(target_group_radios)} options"
    )

    # ============================================================
    # Validate index
    # ============================================================
    idx = int(answer_index) - 1  # Convert to 0-based
    
    if idx < 0 or idx >= len(target_group_radios):
        raise RuntimeError(
            f"Radio button index {answer_index} out of range. "
            f"Group '{target_group_name}' has {len(target_group_radios)} options"
        )

    target_radio = target_group_radios[idx]
    radio_id = target_radio['id']

    logger.info(
        LogCategory.EXECUTION,
        f"[DEBUG] Target radio: id={radio_id}, value={target_radio['value']}"
    )

    # ============================================================
    # CLICK using robust JavaScript approach
    # ============================================================
    click_result = await content_frame.evaluate(
        """(radioId) => {
            const element = document.getElementById(radioId);
            
            if (!element) {
                return {success: false, error: 'Element not found'};
            }

            // Scroll into view (center)
            element.scrollIntoView({behavior: 'instant', block: 'center'});
            
            // Wait a moment for scroll to complete
            return new Promise(resolve => {
                setTimeout(() => {
                    // Check visibility
                    const rect = element.getBoundingClientRect();
                    const isVisible = rect.top >= 0 && 
                                    rect.bottom <= window.innerHeight &&
                                    element.offsetParent !== null;
                    
                    if (!isVisible) {
                        resolve({
                            success: false, 
                            error: 'Element not visible after scroll',
                            rect: {top: rect.top, bottom: rect.bottom}
                        });
                        return;
                    }

                    // Force click and check
                    element.click();
                    element.checked = true;

                    // Trigger change event
                    const event = new Event('change', { bubbles: true });
                    element.dispatchEvent(event);

                    // Also trigger click event (some forms need this)
                    const clickEvent = new Event('click', { bubbles: true });
                    element.dispatchEvent(clickEvent);

                    resolve({
                        success: true, 
                        checked: element.checked,
                        value: element.value
                    });
                }, 500);  // 500ms delay for scroll to finish
            });
        }""",
        radio_id
    )

    logger.info(
        LogCategory.EXECUTION,
        f"[DEBUG] Click result: {click_result}"
    )

    if not click_result.get('success'):
        raise RuntimeError(
            f"Failed to click radio button: {click_result.get('error')}"
        )

    # Verify selection
    await page.wait_for_timeout(1000)

    is_checked = await content_frame.evaluate(
        f"() => document.getElementById('{radio_id}').checked"
    )

    if not is_checked:
        raise RuntimeError(
            f"Radio button id='{radio_id}' was clicked but is not checked"
        )

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] RADIO selected successfully: answer={answer_index}, "
        f"id={radio_id}, value={target_radio['value']}"
    )

    await page.wait_for_timeout(1500)
