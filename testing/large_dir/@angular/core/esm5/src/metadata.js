/**
 * @license
 * Copyright Google Inc. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.io/license
 */
export { Attribute } from './di';
export { ANALYZE_FOR_ENTRY_COMPONENTS, ContentChild, ContentChildren, Query, ViewChild, ViewChildren } from './metadata/di';
export { Component, Directive, HostBinding, HostListener, Input, Output, Pipe } from './metadata/directives';
export { NgModule } from './metadata/ng_module';
export { CUSTOM_ELEMENTS_SCHEMA, NO_ERRORS_SCHEMA } from './metadata/schema';
export { ViewEncapsulation } from './metadata/view';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWV0YWRhdGEuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi8uLi8uLi9wYWNrYWdlcy9jb3JlL3NyYy9tZXRhZGF0YS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7O0dBTUc7QUFjSCxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sTUFBTSxDQUFDO0FBRS9CLE9BQU8sRUFBQyw0QkFBNEIsRUFBRSxZQUFZLEVBQXlCLGVBQWUsRUFBNEIsS0FBSyxFQUFFLFNBQVMsRUFBc0IsWUFBWSxFQUF3QixNQUFNLGVBQWUsQ0FBQztBQUN0TixPQUFPLEVBQUMsU0FBUyxFQUFzQixTQUFTLEVBQXNCLFdBQVcsRUFBd0IsWUFBWSxFQUF5QixLQUFLLEVBQWtCLE1BQU0sRUFBbUIsSUFBSSxFQUFnQixNQUFNLHVCQUF1QixDQUFDO0FBQ2hQLE9BQU8sRUFBbUMsUUFBUSxFQUFvQixNQUFNLHNCQUFzQixDQUFDO0FBQ25HLE9BQU8sRUFBQyxzQkFBc0IsRUFBRSxnQkFBZ0IsRUFBaUIsTUFBTSxtQkFBbUIsQ0FBQztBQUMzRixPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZSBsaWNlbnNlIHRoYXQgY2FuIGJlXG4gKiBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIGF0IGh0dHBzOi8vYW5ndWxhci5pby9saWNlbnNlXG4gKi9cblxuLyoqXG4gKiBUaGlzIGluZGlyZWN0aW9uIGlzIG5lZWRlZCB0byBmcmVlIHVwIENvbXBvbmVudCwgZXRjIHN5bWJvbHMgaW4gdGhlIHB1YmxpYyBBUElcbiAqIHRvIGJlIHVzZWQgYnkgdGhlIGRlY29yYXRvciB2ZXJzaW9ucyBvZiB0aGVzZSBhbm5vdGF0aW9ucy5cbiAqL1xuXG5pbXBvcnQge0F0dHJpYnV0ZX0gZnJvbSAnLi9kaSc7XG5pbXBvcnQge0NvbnRlbnRDaGlsZCwgQ29udGVudENoaWxkcmVuLCBRdWVyeSwgVmlld0NoaWxkLCBWaWV3Q2hpbGRyZW59IGZyb20gJy4vbWV0YWRhdGEvZGknO1xuaW1wb3J0IHtDb21wb25lbnQsIERpcmVjdGl2ZSwgSG9zdEJpbmRpbmcsIEhvc3RMaXN0ZW5lciwgSW5wdXQsIE91dHB1dCwgUGlwZX0gZnJvbSAnLi9tZXRhZGF0YS9kaXJlY3RpdmVzJztcbmltcG9ydCB7RG9Cb290c3RyYXAsIE1vZHVsZVdpdGhQcm92aWRlcnMsIE5nTW9kdWxlfSBmcm9tICcuL21ldGFkYXRhL25nX21vZHVsZSc7XG5pbXBvcnQge1NjaGVtYU1ldGFkYXRhfSBmcm9tICcuL21ldGFkYXRhL3NjaGVtYSc7XG5pbXBvcnQge1ZpZXdFbmNhcHN1bGF0aW9ufSBmcm9tICcuL21ldGFkYXRhL3ZpZXcnO1xuXG5leHBvcnQge0F0dHJpYnV0ZX0gZnJvbSAnLi9kaSc7XG5leHBvcnQge0FmdGVyQ29udGVudENoZWNrZWQsIEFmdGVyQ29udGVudEluaXQsIEFmdGVyVmlld0NoZWNrZWQsIEFmdGVyVmlld0luaXQsIERvQ2hlY2ssIE9uQ2hhbmdlcywgT25EZXN0cm95LCBPbkluaXR9IGZyb20gJy4vaW50ZXJmYWNlL2xpZmVjeWNsZV9ob29rcyc7XG5leHBvcnQge0FOQUxZWkVfRk9SX0VOVFJZX0NPTVBPTkVOVFMsIENvbnRlbnRDaGlsZCwgQ29udGVudENoaWxkRGVjb3JhdG9yLCBDb250ZW50Q2hpbGRyZW4sIENvbnRlbnRDaGlsZHJlbkRlY29yYXRvciwgUXVlcnksIFZpZXdDaGlsZCwgVmlld0NoaWxkRGVjb3JhdG9yLCBWaWV3Q2hpbGRyZW4sIFZpZXdDaGlsZHJlbkRlY29yYXRvcn0gZnJvbSAnLi9tZXRhZGF0YS9kaSc7XG5leHBvcnQge0NvbXBvbmVudCwgQ29tcG9uZW50RGVjb3JhdG9yLCBEaXJlY3RpdmUsIERpcmVjdGl2ZURlY29yYXRvciwgSG9zdEJpbmRpbmcsIEhvc3RCaW5kaW5nRGVjb3JhdG9yLCBIb3N0TGlzdGVuZXIsIEhvc3RMaXN0ZW5lckRlY29yYXRvciwgSW5wdXQsIElucHV0RGVjb3JhdG9yLCBPdXRwdXQsIE91dHB1dERlY29yYXRvciwgUGlwZSwgUGlwZURlY29yYXRvcn0gZnJvbSAnLi9tZXRhZGF0YS9kaXJlY3RpdmVzJztcbmV4cG9ydCB7RG9Cb290c3RyYXAsIE1vZHVsZVdpdGhQcm92aWRlcnMsIE5nTW9kdWxlLCBOZ01vZHVsZURlY29yYXRvcn0gZnJvbSAnLi9tZXRhZGF0YS9uZ19tb2R1bGUnO1xuZXhwb3J0IHtDVVNUT01fRUxFTUVOVFNfU0NIRU1BLCBOT19FUlJPUlNfU0NIRU1BLCBTY2hlbWFNZXRhZGF0YX0gZnJvbSAnLi9tZXRhZGF0YS9zY2hlbWEnO1xuZXhwb3J0IHtWaWV3RW5jYXBzdWxhdGlvbn0gZnJvbSAnLi9tZXRhZGF0YS92aWV3JztcbiJdfQ==