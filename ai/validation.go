package ai

import (
	"encoding/json"
	"fmt"
	"strings"
)

// jsonSchema is the JSON-Schema subset understood by the argument validator.
type jsonSchema = map[string]any

// ValidateToolArguments validates (and coerces) tool-call arguments against a
// tool's JSON-Schema parameters. It returns the coerced argument map or an
// error with a readable path list. It mirrors validateToolArguments from the
// TypeScript ai package minus the TypeBox-only fast path.
func ValidateToolArguments(tool *Tool, args map[string]any) (map[string]any, error) {
	if tool == nil {
		return args, nil
	}
	schema := tool.Parameters
	if schema == nil {
		return args, nil
	}
	coerced, _ := coerceJSONSchemaValue(normalizeOptionalNulls(args, schema), schema)
	if coercedMap, ok := coerced.(map[string]any); ok {
		coerced = coercedMap
	}
	if !matchesJSONSchema(coerced, schema) {
		return nil, fmt.Errorf("validation failed for tool %q:\n%s\n\nreceived arguments:\n%s",
			tool.Name, describeSchemaErrors(coerced, schema), prettyJSON(args))
	}
	out, _ := coerced.(map[string]any)
	if out == nil {
		out = args
	}
	return out, nil
}

func prettyJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}

func describeSchemaErrors(value any, schema jsonSchema) string {
	var out []string
	validateAndCollect(value, schema, "root", &out)
	if len(out) == 0 {
		return "- unknown validation error"
	}
	for i := range out {
		out[i] = "  - " + out[i]
	}
	return strings.Join(out, "\n")
}

// matchesJSONSchema reports whether value satisfies schema.
func matchesJSONSchema(value any, schema jsonSchema) bool {
	return len(validateCollect(value, schema, "root")) == 0
}

func validateCollect(value any, schema jsonSchema, path string) []string {
	var errs []string
	validateAndCollect(value, schema, path, &errs)
	return errs
}

func validateAndCollect(value any, schema jsonSchema, path string, errs *[]string) {
	if schema == nil {
		return
	}
	types := schemaTypes(schema)

	// required
	if req, ok := schema["required"].([]any); ok {
		if obj, isObj := value.(map[string]any); isObj {
			for _, r := range req {
				name, _ := r.(string)
				if _, present := obj[name]; !present {
					*errs = append(*errs, path+"."+name+": is required")
				}
			}
		}
	}

	if len(types) > 0 {
		matched := false
		for _, t := range types {
			if matchesJSONType(value, t) {
				matched = true
				break
			}
		}
		if !matched {
			*errs = append(*errs, fmt.Sprintf("%s: expected %s, got %T", path, strings.Join(types, " or "), value))
			return
		}
	}

	// allOf
	if list, ok := schema["allOf"].([]any); ok {
		for _, sub := range list {
			if s, ok := sub.(map[string]any); ok {
				validateAndCollect(value, s, path, errs)
			}
		}
	}

	// anyOf / oneOf
	handleUnion := func(list []any, exactlyOne bool) {
		matches := 0
		for _, sub := range list {
			s, ok := sub.(map[string]any)
			if !ok {
				continue
			}
			subErrs := validateCollect(value, s, path)
			if len(subErrs) == 0 {
				matches++
			}
		}
		if (exactlyOne && matches != 1) || (!exactlyOne && matches == 0) {
			*errs = append(*errs, path+": does not match any allowed schema")
		}
	}
	if list, ok := schema["anyOf"].([]any); ok {
		handleUnion(list, false)
	}
	if list, ok := schema["oneOf"].([]any); ok {
		handleUnion(list, true)
	}

	// object properties
	if props, ok := schema["properties"].(map[string]any); ok {
		if obj, isObj := value.(map[string]any); isObj {
			for key, ps := range props {
				childSchema, _ := ps.(map[string]any)
				if childSchema == nil {
					continue
				}
				if child, present := obj[key]; present {
					validateAndCollect(child, childSchema, path+"."+key, errs)
				}
			}
		}
	}

	// array items
	if items, ok := schema["items"]; ok {
		if arr, isArr := value.([]any); isArr {
			if itemSchema, ok := items.(map[string]any); ok {
				for i, item := range arr {
					validateAndCollect(item, itemSchema, fmt.Sprintf("%s[%d]", path, i), errs)
				}
			} else if itemList, ok := items.([]any); ok {
				for i, item := range arr {
					if i < len(itemList) {
						if s, ok := itemList[i].(map[string]any); ok {
							validateAndCollect(item, s, fmt.Sprintf("%s[%d]", path, i), errs)
						}
					}
				}
			}
		}
	}
}

func schemaTypes(schema jsonSchema) []string {
	switch t := schema["type"].(type) {
	case string:
		return []string{t}
	case []any:
		var out []string
		for _, x := range t {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func matchesJSONType(value any, typ string) bool {
	switch typ {
	case "number":
		_, ok := value.(float64)
		return ok || isInt(value)
	case "integer":
		if f, ok := value.(float64); ok {
			return f == float64(int(f))
		}
		return isInt(value)
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "string":
		_, ok := value.(string)
		return ok
	case "null":
		return value == nil
	case "array":
		_, ok := value.([]any)
		return ok
	case "object":
		_, ok := value.(map[string]any)
		return ok
	default:
		return false
	}
}

func isInt(v any) bool {
	switch v.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	}
	return false
}

// normalizeOptionalNulls removes null values for optional object properties.
func normalizeOptionalNulls(value any, schema jsonSchema) any {
	if arr, ok := value.([]any); ok {
		items := schema["items"]
		if itemSchema, ok := items.(map[string]any); ok {
			for i := range arr {
				arr[i] = normalizeOptionalNulls(arr[i], itemSchema)
			}
		}
		return arr
	}
	obj, isObj := value.(map[string]any)
	if !isObj {
		return value
	}
	props, _ := schema["properties"].(map[string]any)
	if props == nil {
		return value
	}
	required := map[string]bool{}
	if list, ok := schema["required"].([]any); ok {
		for _, r := range list {
			if name, ok := r.(string); ok {
				required[name] = true
			}
		}
	}
	for key, ps := range props {
		child, present := obj[key]
		if !present {
			continue
		}
		childSchema, _ := ps.(map[string]any)
		if child == nil && !required[key] && childSchema != nil && fetchRef(childSchema) == nil {
			delete(obj, key)
			continue
		}
		if childSchema != nil {
			obj[key] = normalizeOptionalNulls(child, childSchema)
		}
	}
	return obj
}

func fetchRef(s jsonSchema) any {
	if ref, ok := s["$ref"]; ok {
		return ref
	}
	return nil
}

// coerceJSONSchemaValue coerces primitive values toward schema types, then
// recurses into objects and arrays.
func coerceJSONSchemaValue(value any, schema jsonSchema) (any, bool) {
	if schema == nil {
		return value, false
	}
	changed := false
	next := value

	if list, ok := schema["allOf"].([]any); ok {
		for _, sub := range list {
			if s, ok := sub.(map[string]any); ok {
				v, c := coerceJSONSchemaValue(next, s)
				if c {
					changed = true
					next = v
				}
			}
		}
	}
	// union coercion: if any member already matches, keep value
	if list, ok := schema["anyOf"].([]any); ok {
		matched := false
		for _, sub := range list {
			if s, ok := sub.(map[string]any); ok && matchesJSONSchema(next, s) {
				matched = true
				break
			}
		}
		if !matched {
			for _, sub := range list {
				s, ok := sub.(map[string]any)
				if !ok {
					continue
				}
				if v, c := coerceJSONSchemaValue(next, s); c && matchesJSONSchema(v, s) {
					next, changed = v, true
					break
				}
			}
		}
	}

	types := schemaTypes(schema)
	if len(types) > 0 {
		matched := false
		for _, t := range types {
			if matchesJSONType(next, t) {
				matched = true
				break
			}
		}
		if !matched {
			for _, t := range types {
				if v, ok := coercePrimitive(next, t); ok {
					next, changed = v, true
					break
				}
			}
		}
	}

	if arr, ok := next.([]any); ok {
		if items, ok := schema["items"].(map[string]any); ok {
			for i := range arr {
				if v, c := coerceJSONSchemaValue(arr[i], items); c {
					arr[i] = v
					changed = true
				}
			}
		}
	}
	if obj, ok := next.(map[string]any); ok {
		if props, ok := schema["properties"].(map[string]any); ok {
			for key, ps := range props {
				child, present := obj[key]
				if !present {
					continue
				}
				if s, ok := ps.(map[string]any); ok {
					if v, c := coerceJSONSchemaValue(child, s); c {
						obj[key] = v
						changed = true
					}
				}
			}
		}
	}
	return next, changed
}

func coercePrimitive(value any, typ string) (any, bool) {
	switch typ {
	case "number":
		if s, ok := value.(string); ok && strings.TrimSpace(s) != "" {
			var f float64
			if _, err := fmt.Sscanf(s, "%g", &f); err == nil {
				return f, true
			}
		}
		if b, ok := value.(bool); ok {
			if b {
				return float64(1), true
			}
			return float64(0), true
		}
		if isInt(value) {
			return toFloat64(value), true
		}
	case "integer":
		if s, ok := value.(string); ok && strings.TrimSpace(s) != "" {
			var f float64
			if _, err := fmt.Sscanf(s, "%g", &f); err == nil && f == float64(int(f)) {
				return int(f), true
			}
		}
		if b, ok := value.(bool); ok {
			if b {
				return 1, true
			}
			return 0, true
		}
	case "boolean":
		if s, ok := value.(string); ok {
			if s == "true" {
				return true, true
			}
			if s == "false" {
				return false, true
			}
		}
		if f, ok := value.(float64); ok {
			if f == 1 {
				return true, true
			}
			if f == 0 {
				return false, true
			}
		}
	case "string":
		if value == nil {
			return "", true
		}
		if isInt(value) {
			return fmt.Sprintf("%d", value), true
		}
		switch v := value.(type) {
		case float64:
			return fmt.Sprintf("%v", v), true
		case bool:
			if v {
				return "true", true
			}
			return "false", true
		}
	case "null":
		switch value {
		case "", float64(0), 0, false:
			return nil, true
		}
	}
	return value, false
}

func toFloat64(v any) float64 {
	switch n := v.(type) {
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case float64:
		return n
	default:
		return 0
	}
}
