<p align="center"> 
  <img src ="img/banner-light-shafts.jpg" />
</p>

# Writing Plugins for CosmoScout VR
The modularity of CosmoScout allows for easy feature extension through the use of plugins.  
This documentation aims at understanding the basic plugin architecture and the available apis.

## PluginBase
...

## Adding Gui Elements
Elements can be added to the gui by placing them in a `gui` folder in the plugins source.
Structure:
```
csp-example-plugin
└gui
  ├ js
  │ └ plugin.js
  ├ css
  │ └ plugin.css
  └ plugin.html
``` 

### Adding Sidebar Content
Content can be added to the sidebar in two places.

#### Plugin section
Plugins that need extensive configuration or are feature rich should add a designated plugin tab. 
```c++
/// Adds a new tab to the side bar.
///
/// @param name      The nam/title of the tab.
/// @param icon      The name of the Material icon.
/// @param htmlFile  The HTML file that describes the tabs contents.
void addPluginTabToSideBarFromHTML(std::string const& name, std::string const& icon, std::string const& htmlFile);
```

#### Settings section
The settings section is the last tab in the sidebar and contains, as the name denotes, settings.  
Settings should only contain checkboxes for enabling and disabling features or sliders to set ranges. 

GuiManager
```c++
/// Adds a new section to the settings tab.
///
/// @param name      The name/title of the section.
/// @param htmlFile  The HTML file that describes the sections contents.
void addSettingsSectionToSideBarFromHTML(std::string const& name, std::string const& icon, std::string const& htmlFile);
```

### Registering CSS
```c++
/// Adds a link element to the head with a local file href.
///
/// @param fileName The filename in the css folder
void addCssToGui(std::string const& fileName);
```

### Registering JavaScript
```c++
/// This can be used to initialize the DOM elements added to the sidebar with the methods above.
///
/// @param jsFile The javascript file that contains the source code.
void addScriptToGuiFromJS(std::string const& jsFile);
```

## CosmoScout JavaScript API
The global `CosmoScout` object exposes several gui related helper methods.

### `CosmoScout.init(...apis)` 
One or more `IApi` classes to be initialized and registered on the CosmoScout object.  
This method allows you to register and initialize your plugins JavaScript.  
In order to be registerable your api needs to extend the `IApi` interface.  
Registration means that your api is callable as `CosmoScout.apiName.method(args)`.

```javascript
CosmoScout.init(ExampleApi); // One api
CosmoScout.init(FooApi, BarApi, BazApi); // Multiple apis 
```

### `CosmoScout.initDropDowns`
Initializes the `selectpicker` extension to all `.simple-value-dropdown` elements.
A change event listener will be added which calls the CosmoScout application with the elements id and currently selected value.  
This method is idempotent. Event listeners will be only added once.  

```javascript
CosmoScout.initDropDowns();
```

### `CosmoScout.initChecklabelInputs`
Adds a change event listener to all `.checklabel input` elements. On change the CosmoScout application will be called with the elements id and current check state.  
This method is idempotent. Event listeners will be only added once.  

```javascript
CosmoScout.initChecklabelInputs();
```

### `CosmoScout.initRadiolabelInputs`
Adds a change event listener to all `.radiolabel input` elements. On change the CosmoScout application will be called with the elements id.  
This method is idempotent. Event listeners will be only added once.  

```javascript
CosmoScout.initRadiolabelInputs();
```

### `CosmoScout.initDataCalls`
Adds an onclick listener to every element containing `[data-call="'methodname'"]`.  
The method name gets passed to CosmoScout.callNative.  
Arguments can be passed by separating the content with ','  
E.g.: `'fly_to','Africa' -> CosmoScout.callNative('fly_to', 'Africa')`  
      `method,arg1,...,argN -> CosmoScout.callNative('method', arg1, ..., argN)`  
Attribute content will be passed to eval. Strings need to be wrapped in '  
This method is idempotent. Event listeners will be only added once.  

```html
<button data-call="'fly_to', 'Africa'">Fly to Africa</button>
```

```javascript
CosmoScout.initDataCalls();
```

### `CosmoScout.initTooltips`
Initializes all `[data-toggle="tooltip"]` and `[data-toggle="tooltip-bottom"]` tooltips.

```javascript
CosmoScout.initTooltips();
```

### `CosmoScout.initInputs`
This method calls all `init...` methods on the CosmoScout object.

### `CosmoScout.registerJavaScript(url, init)`
Appends a `<script>` element to the body with `url` as its src content. The `init` function gets called on script load.

```javascript
CosmoScout.registerJavaScript('https://example.com/script.js', () => {
    console.log('Script ready');
});
```

### `CosmoScout.unregisterJavaScript(url)`
Removes a registered `<script>` element from the body by its url.

```javascript
CosmoScout.unregisterJavaScript('https://example.com/script.js');
``` 

### `CosmoScout.registerCss(url)`
Appends a `<link rel="stylesheet">` to the head with `url` as its href content.

```javascript
CosmoScout.registerCss('https://example.com/example.css');
```

### `CosmoScout.unregisterCss`
Removes a registered stylesheet by its url.  
Your plugin should call this method upon de-initialization if it added any stylesheets.

```javascript
CosmoScout.unregisterCss('https://example.com/example.css');
```

### `CosmoScout.registerHtml(id, content, containerId = 'body')`
Appends HTML to the body (default) or element with id `containerId`.  
This method gets called by `GuiManager::addHtmlToGui`.

```javascript
const html = '<span>Example Html</span>';

CosmoScout.registerHtml('example', html); // <span> gets appended to the body
CosmoScout.registerHtml('example2', html, 'container') // <span> gets appended to #container
```

### `CosmoScout.unregisterHtml(id, containerId = 'body')`
Remove registered html from the body or container with id `containerId`.  
Your plugin should call this method upon de-initialization if it added any html.

```javascript
CosmoScout.unregisterHtml('example'); // Removes element from body
CosmoScout.unregisterHtml('example2', 'container'); // Removes element from #container
```

### `CosmoScout.loadTemplateContent(templateId)`
In order to avoid mixing Html and JavaScript CosmoScout makes use of `<template>` elements.  
Template elements can contain arbitrary html that won't be displayed and parsed by the browser.  
This allows to add complex html constructs to the GUI without cluttering your JavaScript.  
`CosmoScout.registerHtml` can be used to add `<templates>` to the gui.  

`templateId` will be suffixed by `-template`.  
The return value is either `false` if the template could not be loaded or a `HTMLElement`.

Only the **first** html node of the template will be returned:
```html
<!-- Only the <span> element will be returned -->
<template id="example-template">
    <span>Example</span>
    <p>Example2</p>
</template>
```

```html
<!-- Everything must be wrapped in one element -->
<template id="example2-template">
    <div>
        <span>Example</span>
        <p>Example2</p>
    </div>
</template>
```

```javascript
CosmoScout.loadTemplateContent('example'); // Returns the <span> HTMLElement
CosmoScout.loadTemplateContent('example2'); // Returns the <div> HTMLElement
CosmoScout.loadTemplateContent('example-template'); // Returns false as the method searches for example-template-template
```

### `CosmoScout.clearHtml(element)`
Clears the content of an element if it exists.  
`element` can either be a html id or a HTMLElement.  

```javascript
CosmoScout.clearHtml('container'); // Will clear #container
```

### `CosmoScout.initSlider(id, min, max, step, start)`
Initializes a noUiSlider.
* `id` the sliders html id
* `min` min slider value
* `max` max slider value
* `step` step size
* `start` slider handle count and position

```javascript
CosmoScout.initSlider('set_texture_gamma', 0.1, 3.0, 0.01, [1.0]);
```

### `CosmoScout.setSliderValue(id, ...value)`
Sets the value of a noUiSlider.

```javascript
CosmoScout.setSliderValue('set_texture_gamma', 1); // Value is now 1
CosmoScout.setSliderValue('multi_handle_slider', 1, 2); // First handle value is now 1, second handle is 2
```

### `CosmoScout.clearDropdown(id)`
Clears the content of a dropdown.

```javascript
CosmoScout.clearDropdown('example-dropdown'); // Clears <input type="dropdown" id="example-dropdown">
```

### `CosmoScout.addDropdownValue(id, value, text, selected = false)`
Adds an option element to a dropdown.
* `id` dropdown id
* `value` option value
* `text` option text
* `selected` default false, set to true to add the selected attribute

```javascript
CosmoScout.addDropdownValue('example-dropdpwn', 'value', 'Example option');

CosmoScout.addDropdownValue('example-dropdpwn', 1, 'Example selected', true);
```

### `CosmoScout.setDropdownValue(id, value)`
Sets the current value of a selectpicker.

### `CosmoScout.setRadioChecked(id)`
Sets a radio button to checked.

### `CosmoScout.setCheckboxValue(id, value)`
Sets a checkboxs checked state to true/false.

### `CosmoScout.setTextboxValue(id, value)`
Sets the value of a text input. 
Only selects `.text-input`s which descend `.item-ID`.

### `CosmoScout.callNative(fn, ..args)`
Calls a method on the CosmoScout application.  
`fn` is the applications method name.  
`...args` is a list of arguments.

```javascript
CosmoScout.callNative('fly_to', 'Africa');
CosmoScout.callNative('method', 'arg1', 'arg2', 'argN');
```

### `CosmoScout.register(name, api)`
Called by `init`. Registers an instantiated `IApi` object on the CosmoScout object.  
Makes the registered object accessible as `CosmoScout.name`.

```javascript
const api = new ExampleApi();
CosmoScout.register('exampleApi', api);
```

### `CosmoScout.remove(name)`
Removes a registered api object.

```javascript
CosmoScout.remove('exampleApi');
```

### `CosmoScout.getApi(name)`
Returns a registered api.

## Plugin JavaScript Interface
The `IApi` interface contains a required `name` member which is used by CosmoScout to identify the api.  
The `init` method should contain all code that is needed upon initializing the plugin. This method gets automatically called by the CosmoScout JavaScript api upon initialization.  
```javascript
class IApi {
    // Name of api in camel case
    // Required
    name = "apiName";

    // This gets called upon registration by CosmoScout
    // Optional
    init() {};
}
```

A typical plugin api could look something like this:
```javascript
class ExamplePlugin extends IApi {
    name = 'examplePlugin';

    init() {
        CosmoScout.initSlider('example_slider', 0, 100, 1, 0);
    };

    // This would be callable by the application as:
    // mCosmoScoutGui->callJavascript("CosmoScout.examplePlugin.exampleMethod", arg);
    // And on the JS side as CosmoScout.examplePlugin.exampleMethod(arg);
    // Or internally as this.exampleMethod(arg)
    exampleMethod(arg) {
        //
    }
}

// This ensures plugin initialization on file load
(() => {
    CosmoScout.init(ExamplePlugin);
})();
```


<p align="center"><img src ="img/hr.svg"/></p>

<p align="center">
  <a href="contributing.md">&lsaquo; Contributing to the Project</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
</p>
