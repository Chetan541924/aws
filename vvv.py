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
    
    try:
        await content_frame.wait_for_load_state("networkidle", timeout=15000)
    except:
        logger.warning(LogCategory.EXECUTION, "[DEBUG] Network didn't stabilize, continuing...")
    
    await page.wait_for_timeout(2000)

    # ============================================================
    # STRATEGY: Find ALL radio groups, then intelligently select
    # ============================================================
    
    radio_groups = await content_frame.evaluate("""
        () => {
            const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
            const groups = {};
            
            radios.forEach(radio => {
                const name = radio.name;
                if (!groups[name]) {
                    groups[name] = [];
                }
                
                // Check if element is actually visible
                const rect = radio.getBoundingClientRect();
                const isVisible = (
                    radio.offsetParent !== null &&
                    rect.width > 0 && 
                    rect.height > 0 &&
                    window.getComputedStyle(radio).visibility !== 'hidden' &&
                    window.getComputedStyle(radio).display !== 'none'
                );
                
                groups[name].push({
                    id: radio.id,
                    name: radio.name,
                    value: radio.value,
                    checked: radio.checked,
                    visible: isVisible,
                    top: rect.top,
                    left: rect.left
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
    # SMART SELECTION: Prioritize groups with valid IDs and values
    # ============================================================
    target_group_name = None
    target_group_radios = []

    # Priority 1: Groups with 'answer' in the name (ERBE scenario radios)
    for group_name, radios in radio_groups.items():
        if 'answer' in group_name.lower():
            visible_radios = [r for r in radios if r['visible'] and r['id']]
            if visible_radios and len(visible_radios) >= 2:  # Must have at least 2 options
                target_group_name = group_name
                target_group_radios = visible_radios
                logger.info(
                    LogCategory.EXECUTION,
                    f"[DEBUG] Priority match: Found 'answer' group: {group_name}"
                )
                break

    # Priority 2: Groups with 'customerType' (Customer search radios)
    if not target_group_name:
        for group_name, radios in radio_groups.items():
            if 'customertype' in group_name.lower():
                visible_radios = [r for r in radios if r['visible'] and r['id']]
                if visible_radios:
                    target_group_name = group_name
                    target_group_radios = visible_radios
                    logger.info(
                        LogCategory.EXECUTION,
                        f"[DEBUG] Priority match: Found 'customerType' group: {group_name}"
                    )
                    break

    # Priority 3: Any group with visible radios (excluding filters)
    if not target_group_name:
        for group_name, radios in radio_groups.items():
            # Exclude filter/hidden groups
            if 'filter' in group_name.lower():
                continue
                
            visible_radios = [r for r in radios if r['visible'] and r['id']]
            if visible_radios and len(visible_radios) >= 2:
                target_group_name = group_name
                target_group_radios = visible_radios
                logger.info(
                    LogCategory.EXECUTION,
                    f"[DEBUG] Fallback match: Found group: {group_name}"
                )
                break

    if not target_group_name:
        raise RuntimeError("No visible radio button groups found on page")

    logger.info(
        LogCategory.EXECUTION,
        f"[DEBUG] Selected radio group: {target_group_name} with {len(target_group_radios)} options"
    )

    # Log all radios in the selected group
    for i, radio in enumerate(target_group_radios):
        logger.info(
            LogCategory.EXECUTION,
            f"[DEBUG] Radio {i+1}: id={radio['id']}, value={radio['value']}, checked={radio['checked']}"
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

    if not radio_id:
        raise RuntimeError(f"Radio button at index {answer_index} has no ID")

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
                return {success: false, error: 'Element not found', id: radioId};
            }

            // Scroll into view (center)
            element.scrollIntoView({behavior: 'instant', block: 'center'});
            
            // Wait for scroll to complete
            return new Promise(resolve => {
                setTimeout(() => {
                    // Check visibility
                    const rect = element.getBoundingClientRect();
                    const isVisible = (
                        rect.top >= 0 && 
                        rect.bottom <= window.innerHeight &&
                        element.offsetParent !== null
                    );
                    
                    if (!isVisible) {
                        resolve({
                            success: false, 
                            error: 'Element not visible after scroll',
                            rect: {top: rect.top, bottom: rect.bottom},
                            id: radioId
                        });
                        return;
                    }

                    // Force click and check
                    element.click();
                    element.checked = true;

                    // Trigger events
                    element.dispatchEvent(new Event('change', { bubbles: true }));
                    element.dispatchEvent(new Event('click', { bubbles: true }));

                    resolve({
                        success: true, 
                        checked: element.checked,
                        value: element.value,
                        id: radioId
                    });
                }, 500);
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
            f"Failed to click radio button id='{radio_id}': {click_result.get('error')}"
        )

    # Verify selection
    await page.wait_for_timeout(1000)

    is_checked = await content_frame.evaluate(
        f"() => {{ const el = document.getElementById('{radio_id}'); return el ? el.checked : false; }}"
    )

    if not is_checked:
        raise RuntimeError(
            f"Radio button id='{radio_id}' was clicked but is not checked"
        )

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] RADIO selected successfully: answer={answer_index}, "
        f"group={target_group_name}, id={radio_id}, value={target_radio['value']}"
    )

    await page.wait_for_timeout(1500)
    
    # ✅ ADDED: Continue to next step
    continue
