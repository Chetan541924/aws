elif action_type == "BUTTON":

    if "name =" not in step_name:
        raise RuntimeError("BUTTON step missing button name")

    step_lower = step_name.lower()
    button_name = step_name.split("name =")[-1].strip().lower()

    # Always resolve frames fresh
    nav_frame, content_frame = resolve_ccs_frames(page)

    # Wait for page to be ready
    await content_frame.wait_for_load_state("domcontentloaded", timeout=15000)
    await page.wait_for_timeout(1000)

    # ------------------------------
    # Resolve section explicitly
    # ------------------------------
    section = None
    if " in the " in step_lower:
        section = (
            step_lower.split(" in the ", 1)[1]
            .split(" with name", 1)[0]
            .replace(" section", "")
            .strip()
        )

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Looking for button '{button_name}' in section '{section}'"
    )

    # ------------------------------
    # Locate buttons (with or without section)
    # ------------------------------
    if section:
        # Try to find section container
        section_root = content_frame.locator(
            f"div:has-text('{section.title()}')"
        ).first
        
        # Wait for section to be visible
        try:
            await section_root.wait_for(state="visible", timeout=10000)
        except:
            logger.warning(
                LogCategory.EXECUTION,
                f"Section '{section}' not immediately visible, searching entire frame"
            )
            # Fallback to entire frame if section not found
            section_root = content_frame
        
        buttons = section_root.locator(
            "input[type='button'], input[type='submit'], button"
        )
    else:
        buttons = content_frame.locator(
            "input[type='button'], input[type='submit'], button"
        )
    resolved_button_meta = {
        "value": None,
        "text": None,
        "id": None,
        "index": None
    }

    clicked = False

    count = await buttons.count()
    
    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Found {count} buttons in {'section: ' + section if section else 'frame'}"
    )
    
    if count == 0:
        raise RuntimeError(
            f"No buttons found in {'section: ' + section if section else 'frame'}"
        )

    # ------------------------------
    # ✅ FIXED: Correct indentation
    # ------------------------------
    for i in range(count):
        btn = buttons.nth(i)
        
        # Get button attributes (FIXED INDENTATION)
        value = (await btn.get_attribute("value") or "").lower()
        text = (await btn.text_content() or "").lower()
        id_ = (await btn.get_attribute("id") or "").lower()
        
        # Debug log
        logger.info(
            LogCategory.EXECUTION,
            f"[DEBUG] Button {i+1}: value='{value}', text='{text}', id='{id_}'"
        )

        # Check if this is the button we're looking for
        if button_name in (value + text + id_):
            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] Found matching button: {button_name}"
            )
            
            # Scroll button into view
            await btn.scroll_into_view_if_needed()
            await page.wait_for_timeout(500)  # Let scroll stabilize
            
            # Ensure button is visible and enabled
            await btn.wait_for(state="visible", timeout=10000)
            
            # Hover (helps with some UI frameworks)
            await btn.hover()
            await content_frame.wait_for_timeout(200)
            
            # Click the button
            await btn.click(force=True)
            resolved_button_meta = {
                "value": value,
                "text": text,
                "id": id_,
                "index": i + 1
            }

            
            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] BUTTON clicked: {button_name}"
            )
            
            # ✅ FIXED: Reduced wait from 10s to 3s
            await page.wait_for_timeout(3000)
            
            # ✅ ADDED: Wait for network to stabilize
            try:
                await content_frame.wait_for_load_state("networkidle", timeout=10000)
            except:
                logger.warning(
                    LogCategory.EXECUTION,
                    "Network didn't reach idle state, continuing anyway"
                )
            
            # Additional buffer for UI updates
            await page.wait_for_timeout(1000)
            
            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] BUTTON click successful: {button_name} in {section}"
            )
            
            clicked = True
            break  # Exit the loop once button is clicked

    # ------------------------------
    # ✅ FIXED: Check moved OUTSIDE loop
    # ------------------------------
    if not clicked:
        # ---------------- STEP EXECUTION LOG : BUTTON (FAIL) ----------------
        step_counter += 1
        
        step_execution_logs.append({
            "step_order": step_counter,
            "gherkin_step": step_name,
            "step_type": "WHEN",
            "action_type": "BUTTON",
            "status": "FAIL",
            "error_message": f"Button '{button_name}' not found",
        
            "frame_type": "content",
            "frame_url": content_frame.url if content_frame else None,
        
            "locator_strategy": "semantic_button_match",
            "locator_value": {
                "button_name": button_name,
                "section": section
            },
        
            "resolved_element_tag": None,
            "resolved_element_text": None,
            "resolved_element_id": None,
            "resolved_element_name": None,
            "resolved_element_xpath": None,
        
            "input_value": button_name,
            "confidence": "LOW",
        
            "execution_ts": datetime.now().isoformat()
        })

        raise RuntimeError(
            f"Button '{button_name}' not found in {'section: ' + section if section else 'frame'}"
        )
    
    # ✅ ADDED: Continue to next step
    continue
    # ---------------- STEP EXECUTION LOG : BUTTON ----------------
    step_counter += 1
    
    frame_url = content_frame.url if content_frame else None
    
    step_execution_logs.append({
        "step_order": step_counter,
        "gherkin_step": step_name,
        "step_type": "WHEN",
        "action_type": "BUTTON",
        "status": "PASS",
        "error_message": None,
    
        # Frame details
        "frame_type": "content",
        "frame_url": frame_url,
    
        # Locator strategy
        "locator_strategy": "semantic_button_match",
        "locator_value": {
            "button_name": button_name,
            "section": section,
            "matched_on": resolved_button_meta
        },
    
        # Resolved element metadata
        "resolved_element_tag": "button",
        "resolved_element_text": resolved_button_meta["text"],
        "resolved_element_id": resolved_button_meta["id"],
        "resolved_element_name": None,
        "resolved_element_xpath": None,  # intentionally omitted (dynamic UI)
    
        # Input semantics
        "input_value": button_name,
    
        # Confidence
        "confidence": "HIGH",
    
        "execution_ts": datetime.now().isoformat()
    })

