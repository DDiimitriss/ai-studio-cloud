import io
import sys
import contextlib

def run(args):
    """Execute Python code safely"""
    code = args.get("code", "")
    
    if not code:
        return "❌ No code provided. Tell me what Python code to run!"
    
    # Capture output
    output = io.StringIO()
    
    try:
        # Redirect stdout to capture print statements
        with contextlib.redirect_stdout(output):
            # Create a namespace for execution
            namespace = {}
            exec(code, namespace)
        
        # Get the output
        result = output.getvalue()
        
        # Check if there's a final expression result
        if not result.strip():
            # Try to evaluate the last line as an expression
            lines = code.strip().split('\n')
            if lines:
                try:
                    eval_result = eval(lines[-1], namespace)
                    if eval_result is not None:
                        result = str(eval_result)
                except:
                    pass
        
        if result.strip():
            return f"✅ Code executed successfully!\n\n**Output:**\n```\n{result.strip()}\n```"
        else:
            return "✅ Code executed successfully! (No output)"
            
    except Exception as e:
        return f"❌ Error executing code:\n```\n{str(e)}\n```"

def get_info():
    return {
        "name": "Code Interpreter",
        "description": "Executes Python code and returns the output. Arguments: 'code' (Python code to execute). Use for calculations, data analysis, creating charts, or any Python task."
    }